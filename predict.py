import argparse
import os
import warnings
from enum import Enum

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from eolearn.core import (EOExecutor, EOPatch, EOTask, EOWorkflow, FeatureType,
                          LoadTask, OverwritePermission, SaveTask,
                          linearly_connect_tasks)
from eolearn.io import ExportToTiffTask
from matplotlib.colors import BoundaryNorm, ListedColormap
from sentinelhub import UtmZoneSplitter
from tqdm import tqdm

warnings.filterwarnings('ignore')


class PredictPatchTask(EOTask):
    """
    Задача побудови прогнозу за моделлю на патчі
    Надайте модель і характеристику, а також вихідні назви міток і оцінок
    """

    def __init__(self, model, features_feature, predicted_labels_name, predicted_scores_name=None):
        self.model = model
        self.features_feature = features_feature
        self.predicted_labels_name = predicted_labels_name
        self.predicted_scores_name = predicted_scores_name

    def execute(self, eopatch):
        features = eopatch[self.features_feature]

        t, w, h, f = features.shape
        features = np.moveaxis(features, 0, 2).reshape(w * h, t * f)

        #predicted_labels = self.model.predict(features[:, :171])
        predicted_labels = self.model.predict(features)
        predicted_labels = predicted_labels.reshape(w, h)
        predicted_labels = predicted_labels[..., np.newaxis]
        eopatch[(FeatureType.MASK_TIMELESS, self.predicted_labels_name)] = predicted_labels

        if self.predicted_scores_name:
            #predicted_scores = self.model.predict_proba(features[:, :171])
            predicted_scores = self.model.predict_proba(features)
            _, d = predicted_scores.shape
            predicted_scores = predicted_scores.reshape(w, h, d)
            eopatch[(FeatureType.DATA_TIMELESS, self.predicted_scores_name)] = predicted_scores

        return eopatch


class LULC(Enum):
    """
    Клас Enum, що містить базові типи LULC
    """

    NO_DATA = ("No Data", 0, "#ffffff")
    CULTIVATED_LAND = ("Cultivated Land", 1, "#ffff00")
    FOREST = ("Forest", 2, "#054907")
    GRASSLAND = ("Grassland", 3, "#ffa500")
    SHRUBLAND = ("Shrubland", 4, "#806000")
    WATER = ("Water", 5, "#069af3")
    WETLAND = ("Wetlands", 6, "#95d0fc")
    TUNDRA = ("Tundra", 7, "#967bb6")
    ARTIFICIAL_SURFACE = ("Artificial Surface", 8, "#dc143c")
    BARELAND = ("Bareland", 9, "#a6a6a6")
    SNOW_AND_ICE = ("Snow and Ice", 10, "#000000")

    @property
    def id(self):
        return self.value[1]

    @property
    def color(self):
        return self.value[2]


def main(args):
    EOPATCH_FOLDER = "eopatches_main"
    EOPATCH_SAMPLED_FOLDER = args.sampled_folder if args.sampled_folder else "eopatches_sampled"
    RESULTS_FOLDER = "results"
    id_region_to_be_selected = args.id if args.id else 616

    if args.file:
        # Завантажуємо geojson файл
        geojson = gpd.read_file(args.file)

        # Отримуємо форму країни у форматі полігону
        geojson_data_shape = geojson.geometry.values[0]

        info_list = np.array(UtmZoneSplitter([geojson_data_shape], geojson.crs, 5000).get_info_list())

        patch_ids = []
        for idx, info in enumerate(info_list):
            if abs(info["index_x"] - info_list[id_region_to_be_selected]["index_x"]) <= 2 and abs(
                    info["index_y"] - info_list[id_region_to_be_selected]["index_y"]) <= 2:
                patch_ids.append(idx)

        # Змінити порядок патчів (корисно для відмальовок)
        patch_ids = np.transpose(np.fliplr(np.array(patch_ids).reshape(5, 5))).ravel()
    else:
        patch_ids = joblib.load(os.path.join(RESULTS_FOLDER, "patch_ids.pkl"))

    print("---- 1. Завантажимо наявні EOPatch-i ----")
    load = LoadTask(EOPATCH_SAMPLED_FOLDER)

    print("---- 2. Зробимо прогнозування за допомогою нашої моделі ----")
    model_path = os.path.join(RESULTS_FOLDER, "model_main.pkl")
    model = joblib.load(model_path)
    predict = PredictPatchTask(model, (FeatureType.DATA, "FEATURES"), "LBL_GBM", "SCR_GBM")

    # Збережемо завдання
    save = SaveTask(EOPATCH_SAMPLED_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    print("---- 3. Заекспортимо *.tiff дані ----")
    tiff_location = "results/predicted_tiff"
    os.makedirs(tiff_location, exist_ok=True)
    export_tiff = ExportToTiffTask((FeatureType.MASK_TIMELESS, "LBL_GBM"), tiff_location)

    workflow_nodes = linearly_connect_tasks(load, predict, export_tiff, save)
    workflow = EOWorkflow(workflow_nodes)

    # Створимо список аргументів виконання для кожного патчу
    execution_args = []
    for i in range(len(patch_ids)):
        execution_args.append(
            {
                workflow_nodes[0]: {"eopatch_folder": f"eopatch_{i}"},
                workflow_nodes[2]: {"filename": f"{tiff_location}/prediction_eopatch_{i}.tiff"},
                workflow_nodes[3]: {"eopatch_folder": f"eopatch_{i}"},
            }
        )

    print("---- 4. Запустимо виконання ----")
    executor = EOExecutor(workflow, execution_args)
    executor.run(workers=1, multiprocess=False)
    executor.make_report()

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))

    for i in tqdm(range(len(patch_ids))):
        eopatch_path = os.path.join(EOPATCH_SAMPLED_FOLDER, f"eopatch_{i}")
        eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
        ax = axs[i // 5][i % 5]
        im = ax.imshow(eopatch.mask_timeless["LBL_GBM"].squeeze(),
                       cmap=ListedColormap([x.color for x in LULC], name="lulc_cmap"),
                       norm=BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)],
                                         ListedColormap([x.color for x in LULC], name="lulc_cmap").N))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        del eopatch

    fig.subplots_adjust(wspace=0, hspace=0)

    cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
    cb.ax.tick_params(labelsize=20)
    cb.set_ticks([entry.id for entry in LULC])
    cb.ax.set_xticklabels([entry.name for entry in LULC], rotation=45, fontsize=15)
    plt.savefig(
        f'visualise_the_prediction{"_" + args.file.split("/")[-1].split(".geojson")[0] if args.file else ""}.png')

    # Відмалюємо фактичну карту
    fig = plt.figure(figsize=(15, 15))

    idx = np.random.choice(range(len(patch_ids)))
    inspect_size = 80

    eopatch = EOPatch.load(os.path.join(EOPATCH_SAMPLED_FOLDER, f"eopatch_{idx}"), lazy_loading=True)

    w, h = eopatch.mask_timeless["LULC"].squeeze().shape

    w_min = np.random.choice(range(w - inspect_size))
    w_max = w_min + inspect_size
    h_min = np.random.choice(range(h - inspect_size))
    h_max = h_min + inspect_size

    ax = plt.subplot(2, 2, 1)
    plt.imshow(eopatch.mask_timeless["LBL_GBM"].squeeze()[w_min:w_max, h_min:h_max],
               cmap=ListedColormap([x.color for x in LULC], name="lulc_cmap"),
               norm=BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)],
                                 ListedColormap([x.color for x in LULC], name="lulc_cmap").N))
    plt.xticks([])
    plt.yticks([])
    ax.set_aspect("auto")
    plt.title("Predicted data", fontsize=20)

    ax = plt.subplot(2, 2, 2)
    image = np.clip(eopatch.data["FEATURES"][8][..., [2, 1, 0]] * 3.5, 0, 1)
    plt.imshow(image[w_min:w_max, h_min:h_max])
    ax.set_aspect("auto")
    plt.title("True Color", fontsize=20)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(
        f'visual_inspection_of_patches{"_" + args.file.split("/")[-1].split(".geojson")[0] if args.file else ""}.png')

    print("---- Перевірте файли visualise_the_prediction.png та "
          "visual_inspection_of_patches.png для перегляду результатів ----")

    if args.file == "train_data/svn_border.geojson":
        print("---- Візуалізація довідкової карти ----")
        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))
        for i in tqdm(range(len(patch_ids))):
            eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
            eopatch = EOPatch.load(eopatch_path, lazy_loading=True)

            ax = axs[i // 5][i % 5]
            im = ax.imshow(eopatch.mask_timeless["LULC"].squeeze(),
                           cmap=ListedColormap([x.color for x in LULC], name="lulc_cmap"),
                           norm=BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)],
                                             ListedColormap([x.color for x in LULC], name="lulc_cmap").N))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")
            del eopatch

        fig.subplots_adjust(wspace=0, hspace=0)

        cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
        cb.ax.tick_params(labelsize=20)
        cb.set_ticks([entry.id for entry in LULC])
        cb.ax.set_xticklabels([entry.name for entry in LULC], rotation=45, fontsize=15)
        plt.show();


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Оброблення *.geojson файлу')
    parser.add_argument('--file', type=str, help='Шлях до  *.geojson файлу', default="train_data/svn_border.geojson")
    parser.add_argument('--sampled_folder', type=str, help='Назва папки з засемплованими EOpatch-ами')
    parser.add_argument('--id', type=int, help='Ідентифікатор області для аналізу')

    args = parser.parse_args()
    main(args)
