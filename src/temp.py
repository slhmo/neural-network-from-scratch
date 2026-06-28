import pandas as pd
import numpy as np

from src.NeuralNetworksCuPy import ArtificialNeuralNetworks
from src.utilz import train_test_split

# 4 layer, 784? * n * m * 10 neural network

# col0 = y and each row a sample and other cols correspond to pixels

dataset = pd.read_csv('data/mnist_train.csv', header=None)


train, test = train_test_split(dataset, stratify_over_column=0, test_ratio=0.2) # we want to stratify over y to make sure we have enough of each label in the test set

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

n_train_samples = 5000
x_train = np.array(train.iloc[:n_train_samples, 1:]) / 255.0
y_train = np.array(train.iloc[:n_train_samples, 0])

y_train = np.eye(10)[y_train]       # one hot encoding

ann = ArtificialNeuralNetworks((784, 16, 16, 10), activation_function='relu', final_layer_activation='sigmoid')
ann.fit(x_train, y_train, epochs=50000, learning_rate=0.01)

n_test_samples = 100
x_test = np.array(test.iloc[:n_test_samples, 1:]) / 255.0
y_test = np.array(test.iloc[:n_test_samples, 0])
tmp = ann.predict_probabilities(x_test)

for i in range(5):
    print(f'prediction: {np.argmax(tmp[i])}, label: {y_test[i]}')