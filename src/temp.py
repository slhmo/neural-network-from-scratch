import pandas as pd
import numpy as np

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


