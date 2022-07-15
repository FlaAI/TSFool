# TSFool: Adversarial Time Series Generation Tool based on Interval Weighted Finite Automaton to Fool Recurrent Neural Network Classifiers

July 2022 update: 

- The sample program using TSFool to generate adversarial time series for an LSTM classifier in PowerCons Dataset from UCR Archive is added for reference in advance.

- The work is in progress at present and the detailed description (as well as a possible technology paper) will be opened to the public soon.


## Core Idea

The manifold hypothesis is one of the widely accepted explanations for the effectiveness of NNs. It holds that many high-dimensional datasets in the real world are actually distributed along low-dimensional manifolds embedded in high-dimensional space, which explains why NNs can find potential key features as complex functions of the large number of features in the data and generate accurate predictions. It is through learning the latent manifold of the training data that NNs can realize manifold interpolation between input samples, so as to correctly process and predict unseen samples. As a result, what matters in NN classification is how to distinguish the latent manifolds of samples from different classes instead of the original features. However, due to the limitation of sampling technologies and human cognitive ability, practical label construction and NNs evaluation usually have to rely on specific form of data with external features in the high-dimensional space.

Thus, one of the possible explanations for the existence of the adversarial sample is that, the external representation of the input data does not always have the ability to fully and visually reflect the latent manifold, which makes it possible for samples that are considered to be similar in the external representation to have radically different latent manifolds, and as a result, to be understood and processed in a different way by the NN. Therefore, even a small perturbation in human cognition imposed on the correct sample may completely overturn the NN's view of its latent manifold, so as to result in a completely different result. So if there is a kind of model that can simulate the way an NN understands and processes input data, but distinguish different inputs by their original features in the high-dimensional space just like a human, then it can be used to capture the otherness between the latent manifold and external features of any input sample. And such otherness can serve as a guidance to find the potential vulnerable samples for adversarial attack.

