import argparse
import datetime
import warnings
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from eolearn.core import (EOExecutor, EOTask, EOWorkflow, FeatureType,
                          LoadTask, MergeFeatureTask, OverwritePermission,
                          SaveTask, linearly_connect_tasks, EOPatch)
from eolearn.features import NormalizedDifferenceIndexTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask, VectorImportTask
from sentinelhub import DataCollection, UtmZoneSplitter
from shapely.geometry import Polygon
from eolearn.geometry import ErosionTask
from eolearn.features import LinearInterpolationTask, SimpleFilterTask
from eolearn.ml_tools import FractionSamplingTask

warnings.filterwarnings('ignore')

class ValidDataFractionPredicate:
    """
    Предикат, який визначає, чи є кадр з часового ряду EOPatch дійсним чи ні
    Фрейм є валідним, якщо частка валідних даних перевищує вказаний поріг
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return (np.sum(array.astype(np.uint8)) / np.prod(array.shape)) > self.threshold


class SentinelHubValidDataTask(EOTask):
    """
    Поєднаємо карту класифікації Sen2Cor з `IS_DATA`, щоб визначити маску `VALID_DATA_SH`
    Маску хмари SentinelHub можна знайти у eopatch.mask['CLM']
    """

    def __init__(self, output_feature):
        self.output_feature = output_feature

    def execute(self, eopatch):
        eopatch[self.output_feature] = eopatch.mask["IS_DATA"].astype(bool) & (~eopatch.mask["CLM"].astype(bool))
        return eopatch


class AddValidCountTask(EOTask):
    """
    Підраховує кількість достовірних спостережень у часових рядах і зберігає результати в позачасовій масці
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch[FeatureType.MASK_TIMELESS, self.name] = np.count_nonzero(eopatch.mask[self.what], axis=0)
        return eopatch


def main(args):
    # Папка, в якій зберігаються дані для роботи скрипта
    DATA_FOLDER = "train_data"
    # Папки для зібраних даних і проміжних результатів
    EOPATCH_FOLDER = args.eopatch_folder if args.eopatch_folder else "eopatches_slovenia"
    EOPATCH_SAMPLED_FOLDER = args.sampled_folder if args.sampled_folder else "eopatches_sampled_slovenia"
    RESULTS_FOLDER = "results"

    geojson = args.geojson if args.geojson else "svn_border.geojson"
    ID = args.id if args.id else 616

    print("---- 1. Інформація про GeoJson ----")

    # Завантажуємо geojson файл
    country = gpd.read_file(os.path.join(DATA_FOLDER, geojson))

    # Отримуємо форму у форматі полігону
    country_shape = country.geometry.values[0]

    # Візуалізуємо контури
    country.plot()
    plt.axis("off")
    plt.show()

    # Створимо роздільник, щоб отримати список bbox-ів зі сторонами по 1 км
    bbox_splitter = UtmZoneSplitter([country_shape], country.crs, args.shape)

    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    # Підготуємо інформацію про вибрані EOPatch-і
    geometry = [Polygon(bbox.get_polygon()) for bbox in np.array(bbox_splitter.get_bbox_list())]
    idxs = [info["index"] for info in info_list]
    idxs_x = [info["index_x"] for info in info_list]
    idxs_y = [info["index_y"] for info in info_list]

    bbox_gdf = gpd.GeoDataFrame({"index": idxs, "index_x": idxs_x, "index_y": idxs_y}, crs=country.crs,
                                geometry=geometry)

    # Отримуємо патчі навколо
    # Обираємо область 5x5 (ID - ідентифікатор центральної частини)
    patch_ids = []
    for idx, info in enumerate(info_list):
        if abs(info["index_x"] - info_list[ID]["index_x"]) <= 2 and abs(
                info["index_y"] - info_list[ID]["index_y"]) <= 2:
            patch_ids.append(idx)

    # Змінити порядок патчів (корисно для відмальовок)
    try:
        patch_ids = np.transpose(np.fliplr(np.array(patch_ids).reshape(5, 5))).ravel()
    except ValueError:
        print('Будь-ласка оберіть інший ID області!')
        exit()

    # Збережемо у шейп-файл
    bbox_gdf.to_file(os.path.join(RESULTS_FOLDER, "grid_500x500.gpkg"), driver="GPKG")

    print("---- 2. Візуалізація області, обраної з GeoJSON ----")
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.set_title("Візуалізація області, обраної з GeoJSON", fontsize=25)
    country.plot(ax=ax, facecolor="w", edgecolor="r", alpha=0.5)
    bbox_gdf.plot(ax=ax, facecolor="w", edgecolor="b", alpha=0.5)

    for bbox, info in zip(np.array(bbox_splitter.get_bbox_list()), info_list):
        geo = bbox.geometry
        ax.text(geo.centroid.x, geo.centroid.y, info["index"], ha="center", va="center")

    # Позначимо поля виділеної області
    bbox_gdf[bbox_gdf.index.isin(patch_ids)].plot(ax=ax, facecolor="g", edgecolor="r", alpha=0.5)

    plt.axis("off")
    plt.show()

    # Додамо запит на S2
    # Тут ми також робимо простий фільтр хмарних сцен
    # Маски та ймовірності s2cloudless запитуються за допомогою додаткових даних
    band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
    add_data = SentinelHubInputTask(
        bands_feature=(FeatureType.DATA, "BANDS"),
        bands=band_names,
        resolution=10,
        maxcc=0.8,
        time_difference=datetime.timedelta(minutes=120),
        data_collection=DataCollection.SENTINEL2_L1C,
        additional_data=[(FeatureType.MASK, "dataMask", "IS_DATA"), (FeatureType.MASK, "CLM"),
                         (FeatureType.DATA, "CLP")],
        max_threads=5,
    )

    print('---- 3. Підраховуємо нові фічі ----')
    ndvi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "BANDS"),
        (FeatureType.DATA, "NDVI"),
        [band_names.index("B08"), band_names.index("B04")]
    )
    ndwi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "BANDS"),
        (FeatureType.DATA, "NDWI"),
        [band_names.index("B03"), band_names.index("B08")]
    )
    ndbi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "BANDS"),
        (FeatureType.DATA, "NDBI"),
        [band_names.index("B11"), band_names.index("B08")]
    )

    # Перевіряємо пікселі за допомогою маски виявлення хмар SentinelHub та регіону зйомки
    add_sh_validmask = SentinelHubValidDataTask((FeatureType.MASK, "IS_VALID"))

    # Підраховуємо кількість дійсних спостережень на піксель, використовуючи дійсну маску даних
    add_valid_count = AddValidCountTask("IS_VALID", "VALID_COUNT")

    land_use_ref_path = "train_data/land_use_10class_reference_slovenia.gpkg"

    vector_feature = FeatureType.VECTOR_TIMELESS, "LULC_REFERENCE"

    vector_import_task = VectorImportTask(vector_feature, land_use_ref_path)

    rasterization_task = VectorToRasterTask(
        vector_feature,
        (FeatureType.MASK_TIMELESS, "LULC"),
        values_column="lulcid",
        raster_shape=(FeatureType.MASK, "IS_DATA"),
        raster_dtype=np.uint8,
    )

    save = SaveTask(EOPATCH_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # Визначимо робочий процес
    workflow_nodes = linearly_connect_tasks(
        add_data, ndvi, ndwi, ndbi, add_sh_validmask, add_valid_count, vector_import_task, rasterization_task, save
    )
    workflow = EOWorkflow(workflow_nodes)

    # Часовий інтервал для SH-запиту
    time_interval = [args.start_time_interval, args.finish_time_interval]

    # Визначимо додаткові параметри робочого процесу
    input_node = workflow_nodes[0]
    save_node = workflow_nodes[-1]
    execution_args = []
    for idx, bbox in enumerate(bbox_list[patch_ids]):
        execution_args.append(
            {
                input_node: {"bbox": bbox, "time_interval": time_interval},
                save_node: {"eopatch_folder": f"eopatch_{idx}"},
            }
        )

    # Виконаємо робочий процес
    executor = EOExecutor(workflow, execution_args, save_logs=True)
    executor.run(workers=1)

    print("Завантажимо існуючі EOPatch-і")
    load = LoadTask(EOPATCH_FOLDER)

    print("Зробимо операцію об'єднання фіч")
    concatenate = MergeFeatureTask({FeatureType.DATA: ["BANDS", "NDVI", "NDWI", "NDBI"]}, (FeatureType.DATA, "FEATURES"))

    print("Відфільтруємо хмарні сцени")
    # Збережемо кадри з дійсним покриттям > 85%
    valid_data_predicate = ValidDataFractionPredicate(0.85)
    filter_task = SimpleFilterTask((FeatureType.MASK, "IS_VALID"), valid_data_predicate)

    print("Застосуємо лінійно-часову інтерполяцію")
    # Застосуємо лінійну інтерполяцію повного часового ряду та повторну вибірку дат

    #resampled_range = (args.start_time_interval, args.finish_time_interval, 2)
    resampled_range = ("2019-01-01", "2019-12-31", 20)
    linear_interp = LinearInterpolationTask(
        (FeatureType.DATA, "FEATURES"),  # назва поля для інтерполяції
        mask_feature=(FeatureType.MASK, "IS_VALID"),  # маска для використання в інтерполяції
        copy_features=[(FeatureType.MASK_TIMELESS, "LULC")],  # функції, які потрібно зберегти
        resample_range=resampled_range,
    )

    print("Розмиємо кожен клас опорної карти")
    erosion = ErosionTask(mask_feature=(FeatureType.MASK_TIMELESS, "LULC", "LULC_ERODED"), disk_radius=1)

    print("Проведемо рівномірну вибірку пікселів з патчів")
    spatial_sampling = FractionSamplingTask(
        features_to_sample=[(FeatureType.DATA, "FEATURES", "FEATURES_SAMPLED"), (FeatureType.MASK_TIMELESS, "LULC_ERODED")],
        sampling_feature=(FeatureType.MASK_TIMELESS, "LULC_ERODED"),
        fraction=0.25,  # чверть пунктів
        exclude_values=[0],
    )

    save = SaveTask(EOPATCH_SAMPLED_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # Визначимо робочий процес
    workflow_steps = linearly_connect_tasks(load,
                                            concatenate,
                                            filter_task,
                                            linear_interp,
                                            erosion,
                                            spatial_sampling,
                                            save)

    execution_args = []
    for idx in range(len(patch_ids)):
        execution_args.append(
            {
                workflow_steps[0]: {"eopatch_folder": f"eopatch_{idx}"},  # завантажимо дані з папки
                workflow_steps[-2]: {"seed": 42},  # зробимо вибірку
                workflow_steps[-1]: {"eopatch_folder": f"eopatch_{idx}"},  # збережемо
            }
        )

    executor = EOExecutor(EOWorkflow(workflow_steps), execution_args)
    executor.run(workers=1)

    print('---- Скачування пройшло успішно! ----')
    print('---- Візуалізація аеро-космічного знімку ----')

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
    date = datetime.datetime(2019, 6, 1)
    for i in range(len(patch_ids)):
        eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
        eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
        dates = np.array([timestamp.replace(tzinfo=None) for timestamp in eopatch.timestamps])
        closest_date_id = np.argsort(abs(date - dates))[0]
        ax = axs[i // 5][i % 5]
        ax.imshow(np.clip(eopatch.data["BANDS"][closest_date_id][..., [2, 1, 0]] * 3.5, 0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        del eopatch

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Викачування Sentinel знімків')
    parser.add_argument('--shape', type=int, help='Розмір одного bbox', default=1000)
    parser.add_argument('--geojson', type=str, help='GEOJSON для тренування')
    parser.add_argument('--id', type=int, help='Ідентифікатор області для тренування')
    parser.add_argument('--eopatch_folder', type=str, help='Папка для зберігання EOPatch-ів')
    parser.add_argument('--sampled_folder', type=str, help='Назва папки з засемплованими EOpatch-ами')
    parser.add_argument('--start_time_interval', type=str, help="Початковий час вигрузки", default="2019-01-01")
    parser.add_argument('--finish_time_interval', type=str, help="Кінечний час вигрузки", default="2019-12-31")

    args = parser.parse_args()
    main(args)
