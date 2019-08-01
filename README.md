# ChromaNet

Deep neural networks for chromatin profile prediction.  CS273B Final Project, Stanford University.

In this work, we apply deep neural networks to the problem of {\it de novo} chromatin profile prediction.  Our analysis takes two broad approaches.  First, to model long-term dependencies, we train a purely recurrent neural network.  In particular, a bidirectional-LSTM network was used directly on the sequence, which outperformed a logistic regression baseline.  Secondly, we train a convolutional neural network adapted from the DeepSEA architecture \cite{zhou2015predicting}, to analyze the benefits of multitask learning. We use principal component analysis to identify clusters of tasks, and give evidence that training a network on related tasks improves PR-AUC performance relative to randomly selected tasks.

To run: `bash tf.sh main.py`
