import torch
from pylab import *
import numpy as np
import copy
from WFA import build_WFA, run_WFA, calculate_average_input_distance

torch.manual_seed(1)


def TSFool(model, X, Y, K=2, T=30, F=0.1, eps=0.1, N=20, P=0.9, C=1, target=-1, details=False):
    r"""
        Arguments:
            model (nn.Module): target rnn classifier
            X (numpy.array): time series data (sample_amount, time_step, feature_dim)
            Y (numpy.array): label (sample_amount, )
            K (int): >=1, hyper-parameter of build_WFA(), denotes the number of K-top prediction scores to be considered
            T (int): >=1, hyper-parameter of build_WFA(), denotes the number of partitions of the prediction scores
            F (float): (0, 1], hyper-parameter of build_WFA(), ensures that comparing with the average distance
                        between feature points, the grading size of tokens are micro enough
            eps (float): (0, 0.1], hyper-parameter for perturbation, denotes the perturbation amount under the
                        limitation of 'micro' (value 0.1 corresponds to the largest legal perturbation amount)
            N (int): >=1, hyper-parameter for perturbation, denotes the number of adversarial samples generated from
                     a specific minimum positive sample
            P (float): (0, 1], hyper-parameter for perturbation, denotes the possibility of the random mask
            C (int): >=1, hyper-parameter for perturbation, denote the number of minimum positive samples to be
                     considered for each of the sensitive negative samples
            target (int): [-1, max label], hyper-parameter for perturbation, -1 denotes untargeted attack, other
                          values denote targeted attack with the corresponding label as the target
            details (bool): if True, print the details of the attack process
    """

    # ----------------- Build, Run and Compare WFA and target RNN Model --------------- #

    # build and run WFA
    abst_alphabet, initial_vec, trans_matrices, final_vec = build_WFA(model, X, Y, K, T, F, details)
    wfa_output = run_WFA(X, Y, abst_alphabet, initial_vec, trans_matrices, final_vec)
    rep_model_output = torch.from_numpy(wfa_output)
    rep_model_pred_y = torch.max(rep_model_output, 1)[1].data.numpy()

    # run target model
    X_torch = torch.from_numpy(X).to(torch.float32)
    model_output, _ = model(X_torch)
    model_pred_y = torch.max(model_output, 1)[1].data.numpy()

    # evaluate models similarity
    differ_record = []
    for i_0 in range(X.shape[0]):
        if rep_model_pred_y[i_0] != model_pred_y[i_0]:
            differ_record.append(i_0)

    if details:
        rep_model_accuracy = float((rep_model_pred_y == Y).astype(int).sum()) / float(Y.size)
        print('WFA Accuracy: %.4f' % rep_model_accuracy)
        model_accuracy = float((model_pred_y == Y).astype(int).sum()) / float(Y.size)
        print('Model Accuracy: %.4f' % model_accuracy)
        similarity = (1 - len(differ_record) / X.shape[0]) * 100
        print('\nSimilarity between target RNN classifier and WFA: %.2f' % similarity, '%')

        # calculate the plot range under specific dataset
        ylim_low = 65535
        ylim_high = 0
        for item in X:
            for ts in range(X.shape[1]):
                if item[ts, 0] > ylim_high:
                    ylim_high = item[ts, 0]
                if item[ts, 0] < ylim_low:
                    ylim_low = item[ts, 0]
        input_distance = ylim_high - ylim_low
        plot_distance = (input_distance) / 10

        # plot sensitive negative samples one by one with other samples as background
        timestep_record = [i for i in range(1, X.shape[1] + 1)]
        output_size = int(np.max(Y) + 1)
        for i in differ_record:
            if model_pred_y[i] != Y[i] and rep_model_pred_y[i] == Y[i]:
                color_list = plt.cm.tab10(np.linspace(0, 1, 12))
                for j in range(output_size):
                    plt.plot([], [], color=color_list[j+1], label=f'class {j} samples')
                    for k in range(X.shape[0]):
                        if Y[k] == j:
                            X_record = []
                            for item in np.array(X[k]):
                                X_record.append(item[0])
                            plt.plot(timestep_record, X_record, color=color_list[j+1])
                sns_X_record = []
                for item in np.array(X[i]):
                    sns_X_record.append(item[0])
                plt.plot(timestep_record, sns_X_record, color='red', label='sensitive negative sample')
                plt.ylim((ylim_low - plot_distance / 2, ylim_high + plot_distance / 2))
                plt.title(f'Sensitive Negative Sample: Input Sample No.{i} (Correct Label: {int(Y[i])})')
                plt.legend()
                plt.show()

    # -------- Find Target Positive Samples & Generate Minimum Positive Samples ------- #

    # to make sure the perturbations are 'micro', the perturbation amount is limited to be an order of magnitude
    # smaller than the average distance of input features, with the specific values determined by hyper-parameter eps
    average_input_distance = calculate_average_input_distance(X)
    perturbation_amount = eps * average_input_distance

    target_positive_sample_X = []
    minimum_positive_sample_X = []
    minimum_positive_sample_Y = []
    candidate_perturbation_X = []

    for i in differ_record:
        if model_pred_y[i] != Y[i] and rep_model_pred_y[i] == Y[i]:
            if target != -1 and model_pred_y[i] != target:  # targeted / untargeted attack
                continue

            # order neighbor samples for each of the sensitive negative samples according to abstract distance
            abst_bias = []
            for k in range(X.shape[0]):
                current_abst_bias = 0
                for j in range(X.shape[1]):
                    current_abst_bias += abs(int(abst_alphabet[k, j]) - int(abst_alphabet[i, j]))
                abst_bias.append(current_abst_bias)
            candidate_samples_index = np.argsort(abst_bias)

            current_count = 0
            for candidate_sample_index in candidate_samples_index:

                # find corresponding target positive sample(s) from the candidates
                if model_pred_y[candidate_sample_index] != Y[candidate_sample_index] \
                        or Y[candidate_sample_index] != Y[i]:
                    continue
                current_negative_x = copy.deepcopy(np.array(X[i]))  # initialized as the sensitive negative sample
                current_positive_x = X[candidate_sample_index]  # initialized as the target positive sample

                # iterative sampling between (variable) pos and neg samples
                while True:
                    sampled_instant_X = []
                    sampled_instant_Y = []

                    # build sampler
                    sampling_bias_x = []
                    for j in range(X.shape[1]):
                        sampling_bias_f2 = []
                        for k in range(X.shape[2]):
                            real_bias = current_positive_x[j, k] - current_negative_x[j, k]
                            sampling_bias = real_bias / 10
                            sampling_bias_f2.append(sampling_bias)
                        sampling_bias_x.append(sampling_bias_f2)
                    sampling_bias_x = np.array(sampling_bias_x)

                    # sampling
                    sampled_instant_x = copy.deepcopy(current_negative_x)
                    for j in range(11):
                        sampled_instant_X.append(copy.deepcopy(sampled_instant_x))
                        sampled_instant_Y.append(Y[i])
                        sampled_instant_x += sampling_bias_x

                    # find pos and neg distribution
                    sampled_instant_X = np.array(sampled_instant_X)
                    sampled_instant_Y = np.array(sampled_instant_Y)
                    sampled_instant_X = torch.from_numpy(sampled_instant_X).to(torch.float32)
                    sampled_instant_output, _ = model(sampled_instant_X)
                    pred_sampled_instant_y = torch.max(sampled_instant_output, 1)[1].data.numpy()
                    sampled_instant_acc_record = (pred_sampled_instant_y == sampled_instant_Y)

                    # update sampling range
                    for j in range(len(sampled_instant_acc_record) - 1):
                        if not sampled_instant_acc_record[j] and sampled_instant_acc_record[j + 1]:
                            current_negative_x = copy.deepcopy(np.array(sampled_instant_X[j]))
                            current_positive_x = copy.deepcopy(np.array(sampled_instant_X[j + 1]))

                    # end condition
                    end_flag = True
                    for j in range(X.shape[2]):
                        if sampling_bias_x[:, j].max() > perturbation_amount[j] / 10:
                            end_flag = False
                    if end_flag:
                        target_positive_sample_X.append(X[candidate_sample_index])
                        minimum_positive_sample_X.append(current_positive_x)
                        minimum_positive_sample_Y.append(Y[i])
                        candidate_perturbation_X.append(sampling_bias_x)
                        break

                current_count += 1
                if current_count >= C:
                    break

    # ------ Implement Random Mask Perturbations to Generate Adversarial Samples ------ #

    adv_X = []
    adv_Y = []

    for i in range(len(minimum_positive_sample_X)):
        minimum_positive_sample_x = minimum_positive_sample_X[i]
        minimum_positive_sample_y = minimum_positive_sample_Y[i]
        current_perturbation_x = candidate_perturbation_X[i]

        current_adv_X = []
        for j in range(N):
            current_mask = np.random.choice([0, 1], size=X.shape[1], p=[(1 - P), P]).reshape(-1, 1)
            current_masked_perturbation_x = current_perturbation_x * current_mask
            current_minimum_positive_sample_x = copy.deepcopy(minimum_positive_sample_x)
            current_adv_x = current_minimum_positive_sample_x - current_masked_perturbation_x
            current_adv_X.append(current_adv_x)

        current_adv_Y = [minimum_positive_sample_y for j in range(len(current_adv_X))]
        adv_X.extend(current_adv_X)
        adv_Y.extend(current_adv_Y)

    return np.array(adv_X), np.array(adv_Y), np.array(target_positive_sample_X)


if __name__ == '__main__':
    from models.models_structure.ECG200 import RNN
    model = torch.load('models/ECG200.pkl')
    dataset_name = 'ECG200'
    X = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_X.npy')
    Y = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_Y.npy')
    adv_X, adv_Y, target_X = TSFool(model, X, Y, K=2, T=30, F=0.1, eps=0.01, N=20, P=0.9, C=1, target=-1, details=False)
