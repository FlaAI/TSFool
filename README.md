# TSFool: Crafting Highly-imperceptible Adversarial Time Series through Multi-objective Black-box Attack to Fool RNN Classifiers

## Quick Start

The Python module of TSFool, the pre-trained RNN classifiers
and the crafted adversarial sets are available

<!-- ```python-->
<!-- adv_X, adv_Y, target_X = TSFool(model, X, Y, K=2, T=30, F=0.1, eps=0.01, N=20, P=0.9, C=1, target=-1, details=False)-->
<!-- ```-->


## Update

#### February 2023:
- A new version of TSFool implemented as a Python module is available now.
- The corresponding information would be updated soon (expected to complete before Feb. 9, AOE time).


#### September 2022:
- ~~The raw experiment records about **1) the datasets selected from the UCR archive**, **2) the target LSTM classifiers**, **3) the intervalized weighted finite automatons established in the process**, and **4) the results of the final adversarial attacks** have been opened.~~
- ~~The **experiment code**, the **pre-trained models** and the **crafted adversarial sets** are respectively uploaded in the ``Programs``, ``Models`` and ``UCR-Adv`` for reproducibility as well as to allow verification and possible improvement.~~

#### July 2022:
- ~~The sample program using TSFool to craft adversarial time series for an LSTM classifier in PowerCons Dataset from UCR Archive is added for reference in advance.~~
- ~~The work is in progress at present and the detailed description (as well as a possible technology paper) will be opened to the public soon.~~


## Core Idea

One of the possible explanations for the existence of the adversarial sample is that, the features of the input data cannot always fully and visually reflect the latent manifold, which makes it possible for samples that are considered to be similar in the external features to have radically different latent manifolds, and as a result, to be understood and processed in a different way by the DNN. Therefore, even a small perturbation in human cognition imposed on the correct sample may completely overturn the DNN's view of its latent manifold, so as to result in a completely different result. 

So if there is a kind of representation model that can simulate the way the DNN understands and processes input data, but distinguish different inputs by their original features in the high-dimensional space just like a human, then it can be used to capture the otherness between the latent manifold and external features of any input sample. And such otherness can serve as guidance to find the potentially vulnerable samples for the adversarial attack to improve its success rate, efficiency and quality.

In this project, the Interval Weighted Finite Automaton and Recurrent Neural Network (actually LSTM) are respectively the representation model and the DNN mentioned above. Further transferring this idea to other types of models and data is thought to be feasible tentatively, and such attempts are also in progress at present.


## Detailed Experiment Records

### The 10 Experimental Datasets from UCR Archive

**The UCR Time Series Classification Archive:** https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

| ID     | Type      | Name                           | Train | Test | Class | Length |
|--------|-----------|--------------------------------|-------|------|-------|--------|
| CBF    | Simulated | CBF                            | 30    | 900  | 3     | 128    |
| DPOAG  | Image     | DistalPhalanxOutlineAgeGroup   | 400   | 139  | 3     | 80     |
| DPOC   | Image     | DistalPhalanxOutlineCorrect    | 600   | 276  | 2     | 80     |
| ECG200 | ECG       | ECG200                         | 100   | 100  | 2     | 96     |
| GP     | Motion    | GunPoint                       | 50    | 150  | 2     | 150    |
| IPD    | Sensor    | ItalyPowerDemand               | 67    | 1029 | 2     | 24     |
| MPOAG  | Image     | MiddlePhalanxOutlineAgeGroup   | 400   | 154  | 3     | 80     |
| MPOC   | Image     | MiddlePhalanxOutlineCorrect    | 600   | 291  | 2     | 80     |
| PPOAG  | Image     | ProximalPhalanxOutlineAgeGroup | 400   | 205  | 3     | 80     |
| PPOC   | Image     | ProximalPhalanxOutlineCorrect  | 600   | 291  | 2     | 80     |


### The Results of adversarial attack using TSFool and five common methods on the Experimental Datasets

#### Exp. 1
##### - Dataset: CBF
##### - Original Model Acc (Test Set): 0.7511
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.3311       | 900          | **0.004389**  | 14.12%       | 1.1481     |
| BIM             | 0.7022       | 900          | 0.029421      | 3.17%        | 0.9916     |
| DeepFool        | 0.2911       | 900          | 3.298845      | 12.29%       | 1.0994     |
| PGD             | 0.3311       | 900          | 0.029949      | 14.28%       | 1.1492     |
| Transfer Attack | 0.7422       | 900          | -             | **2.60%**    | 1.0105     |
| TSFool          | **0.2111**   | 720          | 0.042502      | 7.48%        | **0.7425** |

#### Exp. 2
##### - Dataset: DPOAG
##### - Original Model Acc (Test Set): 0.7842
| Method          | Attacked Acc | Generate Num | Time Cost (s) | Perturbation | CC         |
|-----------------|--------------|--------------|---------------|--------------|------------|
| FGSM            | 0.4245       | 139          | **0.00319**   | 64.20%       | 1.1593     |
| BIM             | 0.7554       | 139          | 0.028328      | 13.59%       | 1.1577     |
| DeepFool        | 0.1727       | 139          | 0.961403      | 50.61%       | **1.1297** |
| PGD             | 0.3525       | 139          | 0.02906       | 64.68%       | 1.166      |
| Transfer Attack | 0.8575       | 400          | -             | 9.62%        | 1.8219     |
| TSFool          | **0.1071**   | 140          | 0.036396      | **4.93%**    | 1.6093     |




<!-- ## The Experimental LSTM Classifiers -->

