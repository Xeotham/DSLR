#!../.venv/bin/python
from sys import path
path.append("..")
path.append(".")
from numpy import array, ndarray
from logreg_train_bonus import prepare_dataset, features_name
from dslr_lib.opti_bonus import FeaturesSelector, cross_validation
from dslr_lib.errors import print_error
from dslr_lib.threads_bonus import threaded
from pandas import DataFrame, read_csv
from matplotlib.pyplot import bar, show, hist
from multiprocessing import Manager

def find_opti(matrix_x, matrix_y):
    features_accuracy = []
    accuracy_tol = [0.60, 0.98, 0.98, 0.98, 0.98, 0.98, 0.64, 0.64, 0.64, 0.64, 0.61, 0.25, 0.28]

    for i, tol in enumerate(accuracy_tol):
        f_select = FeaturesSelector(matrix_x, matrix_y, i, tol)
        features_accuracy.append(cross_validation(f_select.find_best_features(), matrix_y))

    features_accuracy = DataFrame(
        data=array(features_accuracy).T,
        dtype=float,
    )
    bar(
        x=range(len(features_accuracy)),
        height=array(features_accuracy),
        tick_label=[f"{i + 1}" for i in range(len(features_accuracy))]
    )
    show()

def hard_features(matrix_x, matrix_y, df):
    features_accuracy = []
    threads = []
    features_list = [
        ["Defense Against the Dark Arts"],
        ['Herbology', 'Defense Against the Dark Arts'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic', 'Potions'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic', 'Potions', 'Care of Magical Creatures'],
        ['Astronomy', 'Herbology', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'History of Magic', 'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'History of Magic', 'Potions', 'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms'],
        ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
         'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms'],
        ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
         'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
         'Flying']
    ]

    @threaded
    def threaded_cv(cv_matrix_x, cv_matrix_y, cv_queue):
        cv_queue.put(cross_validation(cv_matrix_x, cv_matrix_y))

    with Manager() as manager:
        q = manager.Queue()
        for i, names in enumerate(features_list):
            threads.append(threaded_cv(df[names].values, matrix_y, q))
        for thread in threads:
            thread.join()
            features_accuracy.append(q.get())

    bar(
        x=range(len(features_accuracy)),
        height=array(features_accuracy),
        tick_label=[f"{i+1}" for i in range(len(features_accuracy))]
    )
    show()


def main():
    try:
        df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))

        # find_opti(matrix_x, matrix_y)
        hard_features(matrix_x, matrix_y, df)

    except FileNotFoundError:
        print_error("FileNotFoundError: provided file not found.")
    except PermissionError:
        print_error("PermissionError: permission denied on provided file.")
    except EmptyDataError:
        print_error("EmptyDataError: Provided dataset is empty.")
    except KeyError as err:
        print_error(f"KeyError: {err} is not in the required file.")
if __name__ == '__main__':
    main()