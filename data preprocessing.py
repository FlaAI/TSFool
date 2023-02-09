import numpy as np


def ucr_read(dataset_path):
    X = []
    Y = []
    with open(dataset_path, 'r') as f:
        text = f.read()
    lines = text.split('\n')
    for line in lines:
        _list = line.split('\t')
        if len(_list) > 1:
            Y.append(float(_list[0]) - 1)
            flo_list = [[float(num)] for num in _list[1:]]
            X.append(flo_list)
    return X, Y


def label_formatting(Y):
    if -1 not in Y and -2 not in Y:
        return Y
    format_label = max(Y) + 1
    for i in range(len(Y)):
        if Y[i] == -1 or Y[i] == -2:
            Y[i] = format_label
    return Y


def data_preprocessing(dataset_name):
    train_X, train_Y = ucr_read(f'datasets/original/{dataset_name}/{dataset_name}_TRAIN.tsv')
    test_X, test_Y = ucr_read(f'datasets/original/{dataset_name}/{dataset_name}_TEST.tsv')
    train_Y = label_formatting(train_Y)
    test_Y = label_formatting(test_Y)
    np.save(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TRAIN_X.npy', np.array(train_X))
    np.save(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TRAIN_Y.npy', np.array(train_Y))
    np.save(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_X.npy', np.array(test_X))
    np.save(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_Y.npy', np.array(test_Y))


if __name__ == '__main__':
    DATASETS_NAME = ['CBF',
                     'DistalPhalanxOutlineAgeGroup',
                     'DistalPhalanxOutlineCorrect',
                     'ECG200',
                     'GunPoint',
                     'ItalyPowerDemand',
                     'MiddlePhalanxOutlineAgeGroup',
                     'MiddlePhalanxOutlineCorrect',
                     'ProximalPhalanxOutlineAgeGroup',
                     'ProximalPhalanxOutlineCorrect']
    for dataset in DATASETS_NAME:
        data_preprocessing(dataset)
