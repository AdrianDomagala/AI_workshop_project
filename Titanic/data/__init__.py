import itertools
import json
import random
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, List, Iterable, Tuple


from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, list2term, Constant
import copy


import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class TitanicDataset(TorchDataset):
    def __init__(self):
        self.df = pd.read_csv('./data/titanic_dataset.csv')
        self.df_labels = self.df[['Survived']]
        
        self.df = self.df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
        self.df['Sex'] = self.df['Sex'].map({"male": 0, "female": 1})
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        
        self.df_train, self.df_test, self.labels_train, self.labels_test = train_test_split(self.df, self.df_labels, test_size = 0.2)
        
        self.get_train(True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i], int(self.labels[i])
        #return self.dataset[i], self.labels[i]
        
    def get_train(self, train = True):
        if train:
            self.dataset=torch.tensor(self.df_train.to_numpy()).float()
            self.labels=torch.tensor(self.labels_train.to_numpy().reshape(-1)).long()
        else:
            self.dataset=torch.tensor(self.df_test.to_numpy()).float()
            self.labels=torch.tensor(self.labels_test.to_numpy().reshape(-1)).long()
        return self
    
    def get_test(self):
        self.get_train(False)
        return self

titanic_dataset = TitanicDataset()

datasets = {
    "train": copy.deepcopy(titanic_dataset.get_train()),
    "test": copy.deepcopy(titanic_dataset.get_test()),
}


class Titanic_tensor_source(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]


Titanic_train = Titanic_tensor_source("train")
Titanic_test = Titanic_tensor_source("test")


def survive_pred(dataset: str, seed=None):
    pass
    return TitanicOperator(
        dataset_name=dataset,
    )

class TitanicOperator(Dataset, TorchDataset):

    def __init__( self, dataset_name: str, seed=None):
        super(TitanicOperator, self).__init__()

        self.dataset_name = dataset_name
        self.data = datasets[self.dataset_name]
        self.function_name = "if_survive"

    def __getitem__(self, index: int) -> Tuple[list, int]:
        l1 = self.data[index][0]
        label = self.data[index][1]
        return l1, label

    def to_query(self, i: int) -> Query:
        x, y = self.data[i]
        return Query(Term(self.function_name, Term("a"), Term(y)))

    def __len__(self):
        return len(self.data)
