import itertools
import argparse
import os
import warnings
from enum import Enum

import geopandas as gpd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from eolearn.core import EOPatch
from sentinelhub import UtmZoneSplitter
from sklearn import metrics

warnings.filterwarnings('ignore')


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


def visualise_matrix(matrix, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues,
                     ylabel="True label", xlabel="Predicted label"):
    """
    Ця функція будує матрицю помилок
    Нормалізацію можна застосувати, встановивши `normalize=True`
    """
    np.set_printoptions(precision=2, suppress=True)

    if normalize:
        normalisation_factor = matrix.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps
        matrix = matrix.astype("float") / normalisation_factor

    plt.imshow(matrix, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=18)

    plt.xticks(np.arange(len(classes)), classes, rotation=90, fontsize=18)
    plt.yticks(np.arange(len(classes)), classes, fontsize=18)

    threshold = matrix.max() / 2.0
    for item, j in itertools.product(range(matrix.shape[0]),
                                     range(matrix.shape[1])):
        plt.text(
            j,
            item,
            format(matrix[item, j], ".2f" if normalize else "d"),
            horizontalalignment="center",
            color="white" if matrix[item, j] > threshold else "black",
            fontsize=12,
        )

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)


def main(args):
    # Папка, в якій зберігаються дані для роботи скрипта
    DATA_FOLDER = "train_data"
    # Папки для зібраних даних і проміжних результатів
    EOPATCH_FOLDER, EOPATCH_SAMPLED_FOLDER, RESULTS_FOLDER = "eopatches_main", "eopatches_sampled_main", "results"
    ID = args.id if args.id else 616

    print("---- 1. Інформація про країну ----")

    # Завантажуємо geojson файл
    country_data = gpd.read_file(os.path.join(DATA_FOLDER, "svn_border.geojson")).buffer(500)

    # Отримуємо форму країни у форматі полігону
    country_data_shape = country_data.geometry.values[0]

    # Візуалізуємо контури країни
    country_data.plot()
    plt.axis("off")
    plt.show()

    # Виведемо розмір площі
    print(f"Ширина становить {int(country_data_shape.bounds[2] - country_data_shape.bounds[0])} метрів")
    print(f"Висота становить {int(country_data_shape.bounds[3] - country_data_shape.bounds[1])} метрів")
    print(
        f"Площа становить {int(country_data_shape.bounds[2] - country_data_shape.bounds[0]) * int(country_data_shape.bounds[3] - country_data_shape.bounds[1])} м^2")
    print()

    info_list = np.array(UtmZoneSplitter([country_data_shape], country_data.crs, 5000).get_info_list())

    # Отримуємо патчі навколо
    # Обираємо область 5x5 (ID - ідентифікатор центральної частини)
    patch_ids = []
    for idx, info in enumerate(info_list):
        if abs(info["index_x"] - info_list[ID]["index_x"]) <= 2 and abs(info["index_y"] - info_list[ID]["index_y"]) <= 2:
            patch_ids.append(idx)

    print("---- 2. Побудуємо модель та виконаємо її навчання ----")

    # Завантажимо вибірку eopatch-ів
    sampled_eopatches = []

    for i in range(len(patch_ids)):
        sampled_eopatches.append(EOPatch.load(os.path.join(EOPATCH_SAMPLED_FOLDER, f"eopatch_{i}"), lazy_loading=True))

    # Визначимо ідентифікатори тренувальних та тестових патчів, візьмемо 80% для тренувальної вибірки
    test_ids = [0, 8, 16, 19, 20]
    test_eopatches = [sampled_eopatches[i] for i in test_ids]
    train_ids = [i for i in range(len(patch_ids)) if i not in test_ids]
    train_eopatches = [sampled_eopatches[i] for i in train_ids]

    # Встановимо фічі та мітки для тренувальних і тестових наборів даних
    features_train = np.concatenate([eopatch.data["FEATURES_SAMPLED"] for eopatch in train_eopatches], axis=1)
    labels_train = np.concatenate([eopatch.mask_timeless["LULC_ERODED"] for eopatch in train_eopatches], axis=0)

    features_test = np.concatenate([eopatch.data["FEATURES_SAMPLED"] for eopatch in test_eopatches], axis=1)
    labels_test = np.concatenate([eopatch.mask_timeless["LULC_ERODED"] for eopatch in test_eopatches], axis=0)

    # Отримуємо розмірність
    t, w1, h, f = features_train.shape
    _, w2, _, _ = features_test.shape

    # Змінимо форму до n*m
    features_train = np.moveaxis(features_train, 0, 2).reshape(w1 * h, t * f)
    labels_train = labels_train.reshape(w1 * h)

    features_test = np.moveaxis(features_test, 0, 2).reshape(w2 * h, t * f)
    labels_test = labels_test.reshape(w2 * h)

    # Організуємо лейбли для навчання
    labels_unique = np.unique(labels_train)

    print(labels_unique)

    # Засетапимо модель
    model = lgb.LGBMClassifier(objective="multiclass",
                               num_class=len(labels_unique),
                               metric="multi_logloss")

    # Тренуємо модель
    model.fit(features_train, labels_train)

    # Збережемо модель
    joblib.dump(model, os.path.join(RESULTS_FOLDER, "model_main1.pkl"))

    # Зробимо прогноз по тестовій вибірці
    predicted_labels_test = model.predict(features_test)

    class_labels = np.unique(labels_test)
    class_names = [lulc_type.name for lulc_type in LULC]
    mask = np.in1d(predicted_labels_test, labels_test)
    predictions = predicted_labels_test[mask]
    true_labels = labels_test[mask]

    # Виведемо основні метрики
    f1_scores = metrics.f1_score(true_labels, predictions, labels=class_labels, average=None)
    avg_f1_score = metrics.f1_score(true_labels, predictions, average="weighted")
    recall = metrics.recall_score(true_labels, predictions, labels=class_labels, average=None)
    precision = metrics.precision_score(true_labels, predictions, labels=class_labels, average=None)
    accuracy = metrics.accuracy_score(true_labels, predictions)

    print(f"Точність класифікації: {round(100 * accuracy, 2)} %")
    print(f"F1-score класифікації: {round(100 * avg_f1_score, 2)} %")
    print()
    print("    Клас              =>  F1  | precision | recall")
    print("--------------------------------------------------")
    for idx, lulctype in enumerate([class_names[idx] for idx in class_labels]):
        print(" {0:20s} => {1:2.1f} |  {2:2.1f}  | {3:2.1f}".format(lulctype,
                                                                    f1_scores[idx] * 100,
                                                                    recall[idx] * 100,
                                                                    precision[idx] * 100))
    print()

    print("---- 3. Побудуємо стандартну та транспоновану матрицю помилок ----")

    fig = plt.figure(figsize=(20, 20))

    plt.subplot(1, 2, 1)
    visualise_matrix(
        metrics.confusion_matrix(true_labels, predictions),
        classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
        normalize=True,
        ylabel="Truth (LAND COVER)",
        xlabel="Predicted (GBM)",
        title="Confusion matrix",
    )

    plt.subplot(1, 2, 2)
    visualise_matrix(
        metrics.confusion_matrix(predictions, true_labels),
        classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
        normalize=True,
        xlabel="Truth (LAND COVER)",
        ylabel="Predicted (GBM)",
        title="Transposed Confusion matrix",
    )

    fig.tight_layout()
    plt.savefig('confusion_matrix.png')

    fig = plt.figure(figsize=(30, 5))

    label_ids, label_counts = np.unique(labels_train, return_counts=True)

    plt.bar(range(len(label_ids)), label_counts)
    plt.xticks(range(len(label_ids)), [class_names[i] for i in label_ids], rotation=45, fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('labels_visualisation.png', bbox_inches="tight")

    print("---- Кінець тренування ----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Тренування моделі')
    parser.add_argument('--id', type=int, help='Ідентифікатор області для тренування')

    args = parser.parse_args()
    main(args)
