import pandas as pd
import numpy as np

from src.NeuralNetworkSimple import ArtificialNeuralNetworks

# 4 layer, 784? * n * m * 10 neural network

# col0 = y and each row a sample and other cols correspond to pixels

dataset = pd.read_csv('data/mnist_train.csv', header=None)

# stable stratified splliting!!
# ultimate sepperation!!
from zlib import crc32

def in_test_set(test_ratio=0.2):
    # Key must be stable across runs and updates
    keys = dataset[0].astype(str) + "|" + dataset.index.astype(str)     # key composed of index(unique, to hold stable) and value(stratified)
    return np.array([crc32(key.encode('utf-8')) & 0xffffffff for key in keys]) < test_ratio * 2**32


g_test = in_test_set(0.2)
train = dataset[~g_test].reset_index(drop=True)
test = dataset[g_test].reset_index(drop=True)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

n_train_samples = 5000
x_train = np.array(train.iloc[:n_train_samples, 1:]) / 255.0
y_train = np.array(train.iloc[:, 0])

y_train = np.eye(10)[y_train]       # one hot encoding

ann = ArtificialNeuralNetworks((784, 16, 16, 10))
ann.fit(x_train, y_train, activation_function='sigmoid')

n_test_samples = 100
x_test = np.array(test.iloc[:n_test_samples, 1:]) / 255.0
y_test = np.array(test.iloc[:n_test_samples, 0])
tmp = ann.predict_probabilities(x_test, 'sigmoid')

for i in range(5):
    print(f'prediction: {np.argmax(tmp[i])}, label: {y_test[i]}')