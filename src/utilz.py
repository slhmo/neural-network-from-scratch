import numpy as np
from zlib import crc32
# stable stratified splitting!!
# ultimate separation!!
def train_test_split(dataset, stratify_over_column, test_ratio=0.2):
    """
    :param stratify_over_column: column index containing values which we want to stratify over
    :return: train and test datasets
    """
    i = stratify_over_column
    def in_test_set():
        # Key must be stable across runs and updates
        keys = dataset[i].astype(str) + "|" + dataset.index.astype(str)  # key composed of index(unique, to hold stable) and column we want to stratify over
        return np.array([crc32(key.encode('utf-8')) & 0xffffffff for key in keys]) < test_ratio * 2 ** 32

    g_test = in_test_set()
    train = dataset[~g_test].reset_index(drop=True)
    test = dataset[g_test].reset_index(drop=True)
    return train, test

    # print(f"Train shape: {train.shape}, Test shape: {test.shape}")