import torch
import numpy as np
import time
from main import TSFool


if __name__ == '__main__':
    from models.models_structure.CBF import RNN
    model = torch.load('models/CBF.pkl')
    dataset_name = 'CBF'
    X = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_X.npy')
    Y = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_Y.npy')
    time_start = time.time()
    adv_X, adv_Y, target_X = TSFool(model, X, Y, K=2, T=30, F=0.1, eps=0.1, N=20, P=0.9, C=1, target=-1, details=False)
    time_end = time.time()
    np.save(f'datasets/adversarial/{dataset_name}/{dataset_name}_TEST_ADV_X.npy', adv_X)
    np.save(f'datasets/adversarial/{dataset_name}/{dataset_name}_TEST_ADV_Y.npy', adv_Y)

    # original accuracy
    X_torch = torch.from_numpy(X).to(torch.float32)
    output, _ = model(X_torch)
    model_pred_y = torch.max(output, 1)[1].data.numpy()
    model_accuracy = float((model_pred_y == Y).astype(int).sum()) / float(Y.size)

    # attacked accuracy
    adv_X_torch = torch.from_numpy(adv_X).to(torch.float32)
    adv_output, _ = model(adv_X_torch)
    adv_model_pred_y = torch.max(adv_output, 1)[1].data.numpy()
    adv_model_accuracy = float((adv_model_pred_y == adv_Y).astype(int).sum()) / float(adv_Y.size)

    # average perturbation amount (percentage)
    sum_perturbation_amount = 0
    group_size = int(adv_X.shape[0] / target_X.shape[0])
    for i in range(target_X.shape[0]):
        current_target_x = target_X[i]
        for j in range(group_size):
            current_adv_x = adv_X[i * group_size + j]
            for k in range(target_X.shape[1]):
                sum_perturbation_amount += abs(current_target_x[k][0]-current_adv_x[k][0])
    avg_perturbation_amount = sum_perturbation_amount / len(adv_X)

    max_x = np.array([-65535 for i in range(X.shape[1])])
    min_x = np.array([65535 for i in range(X.shape[1])])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j][0] > max_x[j]:
                max_x[j] = X[i][j][0]
            if X[i][j][0] < min_x[j]:
                min_x[j] = X[i][j][0]
    range_amount = sum(max_x - min_x)
    avg_perturbation_amount_percentage = 100 * (avg_perturbation_amount / range_amount)

    # average camouflage coefficient
    classes_mean = []
    for i in range(int(np.max(Y) + 1)):
        current_sample_number = 0
        current_sample_sum = np.zeros((X.shape[1], X.shape[2]))
        for j in range(X.shape[0]):
            if Y[j] == i:
                current_sample_number += 1
                for k in range(X.shape[1]):
                    current_sample_sum[k][0] += X[j][k][0]
        current_sample_mean = current_sample_sum / current_sample_number
        classes_mean.append(current_sample_mean)
    classes_distance_mean = []
    for i in range(int(np.max(Y) + 1)):
        current_sample_number = 0
        current_classes_distance_sum = 0
        current_classes_mean = classes_mean[i]
        for j in range(X.shape[0]):
            if Y[j] == i:
                current_sample_number += 1
                for k in range(X.shape[1]):
                    current_classes_distance_sum += abs(X[j][k][0] - current_classes_mean[k][0])
        current_classes_distance_mean = current_classes_distance_sum / current_sample_number
        classes_distance_mean.append(current_classes_distance_mean)
    camouflage_coefficient_count = 0
    sum_camouflage_coefficient = 0
    for i in range(adv_X.shape[0]):
        if adv_model_pred_y[i] != adv_Y[i]:
            camouflage_coefficient_count += 1
            current_classes_distance_original = classes_distance_mean[int(adv_Y[i])]
            current_classes_distance_adv = classes_distance_mean[int(adv_model_pred_y[i])]
            current_classes_mean_original = classes_mean[int(adv_Y[i])]
            current_classes_mean_adv = classes_mean[int(adv_model_pred_y[i])]
            current_distance_original = 0
            current_distance_adv = 0
            for j in range(adv_X.shape[1]):
                current_distance_original += abs(adv_X[i][j][0] - current_classes_mean_original[j][0])
                current_distance_adv += abs(adv_X[i][j][0] - current_classes_mean_adv[j][0])
            sum_camouflage_coefficient += (current_distance_original / current_classes_distance_original) / \
                                          (current_distance_adv / current_classes_distance_adv)
    avg_camouflage_coefficient = sum_camouflage_coefficient / camouflage_coefficient_count

    print('\nTSFool Attack:')
    print(f' - Dataset: UCR-{dataset_name}')
    print(' - Model Accuracy [original]: %.4f' % model_accuracy)
    print(' - Model Accuracy [attacked]: %.4f' % adv_model_accuracy)
    print(f' - The Number of Generated Adversarial Samples: {len(adv_X)}')
    print(' - Average Time Cost (per sample): %fS' % ((time_end - time_start) / len(adv_X)))
    print(' - Average Perturbation Amount (L2 distance in percentage): %.2f%%' % avg_perturbation_amount_percentage)
    print(' - Average Camouflage Coefficient: %.4f' % avg_camouflage_coefficient)


