import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import heapq
from random import randrange
from random import seed as set_seed
import numpy as np
from numba import njit, prange
from pandas.api.types import is_numeric_dtype

import numpy as np
import pandas as pd
import torch.utils.data

import torch
from torch import Tensor, nn
import time

from scipy.sparse.linalg import svds


class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        self.data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :4]
        self.items = self.data[:, :2].astype(np.int32) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(self.data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.int64)
        self.item_field_idx = np.array((1,), dtype=np.int64)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 0] = 0
        target[target > 0] = 1
        return target


class MovieLens1MDataset(MovieLens20MDataset):
    """
    MovieLens 1M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path, sep='::', engine='python', header=None)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.items = self.data[:, :2].astype(np.int32) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(self.data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.int64)
        self.item_field_idx = np.array((1,), dtype=np.int64)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 0] = 0
        target[target > 0] = 1
        return target


def rr(
    gt_items,
    predictions,
    topn = None
) -> float:
    '''
    Reciprocal Rank (RR) metric
    '''
    if topn is None:
        topn = len(predictions)

    predictions = np.array(predictions[:topn])
    gt_items = set(gt_items)
    relevance = np.isin(predictions, list(gt_items)).astype(int)
    first_relevant_index = np.argmax(relevance) if np.any(relevance) else -1

    if first_relevant_index == -1:
        return 0.0
    return 1.0 / (first_relevant_index + 1)

def hr(
    gt_items,
    predictions,
    topn = None
) -> float:
    '''
    Hit Rate (HR) metric
    '''
    if topn is None:
        topn = len(predictions)

    predictions = np.array(predictions[:topn])
    gt_items = set(gt_items)
    relevance = np.isin(predictions, list(gt_items)).astype(int)
    return int(np.any(relevance))

def ndcg(
    gt_items,
    predictions,
    topn=None
) -> float:
    '''
    Normalized Discounted Cumulative Gain (NDCG) metric
    '''
    if topn is None:
        topn = len(predictions)

    predictions = np.array(predictions[:topn])
    gt_items = set(gt_items)
    
    relevance = np.isin(predictions, list(gt_items)).astype(int)
    
    dcg = 0.0
    for i, rel in enumerate(relevance, 1):
        dcg += rel / np.log2(i + 1)
    ideal_relevance = np.ones(min(len(gt_items), topn))
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance, 1):
        idcg += rel / np.log2(i + 1)
    ndcg_score = dcg / idcg if idcg > 0 else 0.0
    
    return ndcg_score

def metrics(targets, preds):
    ndcg1 = []
    ndcg5 = []
    ndcg10 = []
    hr1 = []
    hr5 = []
    hr10 = []
    cov = set()
    for u in tqdm(targets.keys()):
        tar = targets[u]
        pr = preds[u]
        cov.add(pr[0])
        for t in tar:
            t = np.array([t])
            hr1 += [hr(t, pr, 1)]
            hr5 += [hr(t, pr, 5)]
            hr10 += [hr(t, pr, 10)]
            ndcg1 += [ndcg(t, pr, 1)]
            ndcg5 += [ndcg(t, pr, 5)]
            ndcg10 += [ndcg(t, pr, 10)]
        
    return {
        "ndcg@1": np.mean(ndcg1),
        "ndcg@5": np.mean(ndcg5),
        "ndcg@10": np.mean(ndcg10),
        "hits@1": np.mean(hr1),
        "hits@5": np.mean(hr5),
        "hits@10": np.mean(hr10),
        "cov":  len(cov) / item_num}



def convert(data, target, prediction):
    '''
    Converts list of targets and predictions to the dict with target and
    prediction for each user.
    '''
    preds = dict()
    targets = dict()
    for i in tqdm(range(len(data))):
        if data[i][0] in preds:
            preds[data[i][0]] += [prediction[i]]
            targets[data[i][0]] += [target[i]]
        else:
            preds[data[i][0]] = [prediction[i]]
            targets[data[i][0]] = [target[i]]
    for u in tqdm(preds.keys()):
        inds = np.argsort(preds[u])[::-1]
        preds[u] = inds
        targets[u] = np.array([i for i in inds if targets[u][i] > 0.5])
    return preds, targets


def sampler_neg(field_embs, items_emb, n):
    """
    Negative sampler
    """
    matrix = torch.mm(torch.sqrt(1 + torch.sum(field_embs ** 2, dim = 1)).unsqueeze(1), torch.sqrt(1 + torch.sum(items_emb ** 2, dim = 1)).unsqueeze(0)) - torch.mm(field_embs, items_emb.transpose(0, 1))
    n_items = items_emb.shape[0]
    z = -torch.log(-torch.log(torch.rand_like(matrix)))
    return (matrix +  z).argmin(dim=1)
