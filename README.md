# TSFool: Adversarial Time Series Generation Tool based on Interval Weighted Finite Automaton to Fool Recurrent Neural Network Classifiers

July 2022 update: 

- The sample program using TSFool to generate adversarial time series for an LSTM classifier in PowerCons Dataset from UCR Archive is added for reference in advance.

- The work is in progress at present and the detailed description (as well as a possible technology paper) will be opened to the public soon.


## Core Idea

One of the possible explanations for the existence of the adversarial sample is that, the features of the input data cannot always fully and visually reflect the latent manifold, which makes it possible for samples that are considered to be similar in the external features to have radically different latent manifolds, and as a result, to be understood and processed in a different way by the NN. Therefore, even a small perturbation in human cognition imposed on the correct sample may completely overturn the NN's view of its latent manifold, so as to result in a completely different result. 

So if there is a kind of model that can simulate the way an NN understands and processes input data, but distinguish different inputs by their original features in the high-dimensional space just like a human, then it can be used to capture the otherness between the latent manifold and external features of any input sample. And such otherness can serve as guidance to find the potential vulnerable samples for the adversarial attack to improve its success rate, efficiency and quality.

In this project, Interval Weighted Finite Automaton and Recurrent Neural Network (actually LSTM) are respectively the model and the NN mentioned above.
