import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from sktime.utils import load_data
import warnings
import json

warnings.filterwarnings('ignore')




class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class YiDongLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        time_start_index = -5
        self.time_start_index = time_start_index
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "train_data.npy"))
        data_wo_time = data[:, :, :time_start_index]
        self.entity_num = data.shape[0]

        self.scaler.fit(data_wo_time.reshape(-1, data_wo_time.shape[-1]))
        data_wo_time = self.scaler.transform(data_wo_time.reshape(-1, data_wo_time.shape[-1]))
        test_data = np.load(os.path.join(root_path, "test_data.npy"))
        test_data_wo_time = test_data[:, :, :time_start_index]
        test_data_wo_time = self.scaler.transform(test_data_wo_time.reshape(-1, test_data_wo_time.shape[-1])).reshape(self.entity_num, -1, test_data_wo_time.shape[-1])
        self.test = np.concatenate([test_data_wo_time, test_data[:, :, time_start_index:]], axis=-1)
        
        data_wo_time = data_wo_time.reshape(self.entity_num, -1, data_wo_time.shape[-1])
        data = np.concatenate([data_wo_time, data[:, :, time_start_index:]], axis=-1)
        self.train = data
        
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "label.npy"))
        print("test:", self.test.shape, "test_labels: ", self.test_labels.shape)
        print("train:", self.train.shape)

    def __len__(self):
        # if self.flag == "train":
        #     return ((self.train.shape[1] - self.win_size) // self.step + 1) * self.train.shape[0]
        # elif (self.flag == 'val'):
        #     return ((self.val.shape[1] - self.win_size) // self.step + 1) * self.val.shape[0]
        # elif (self.flag == 'test'):
        #     return ((self.test.shape[1] - self.win_size) // self.step + 1) * self.test.shape[0]
        
        if self.flag == "train":
            return ((self.train.shape[1] - self.win_size +1 )// self.step) * self.train.shape[0]
        elif (self.flag == 'val'):
            return ((self.val.shape[1] - self.win_size  + 1) // self.step ) * self.val.shape[0]
        elif (self.flag == 'test'):
            return ((self.test.shape[1] - self.win_size  + 1) // self.step) * self.test.shape[0]
        

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            outter_index = index // ((self.train.shape[1] - self.win_size + 1) // self.step )
            inner_index = index % ((self.train.shape[1] - self.win_size + 1) // self.step )
            return np.float32(self.train[outter_index, inner_index:inner_index + self.win_size, :self.time_start_index]), np.float32(
                self.test_labels[outter_index, 0:0 + self.win_size]), np.float32(self.train[outter_index, inner_index:inner_index + self.win_size, self.time_start_index:])
        elif (self.flag == 'val'):
            outter_index = index // ((self.val.shape[1] - self.win_size + 1) // self.step)
            inner_index = index % ((self.val.shape[1] - self.win_size + 1) // self.step)
            return np.float32(self.val[outter_index, inner_index:inner_index + self.win_size, :self.time_start_index]), np.float32(
                self.test_labels[outter_index, 0:0 + self.win_size]), np.float32(self.val[outter_index, inner_index:inner_index + self.win_size, self.time_start_index:])
        elif (self.flag == 'test'):
            outter_index = index // ((self.test.shape[1] - self.win_size + 1) // self.step )
            inner_index = index % ((self.test.shape[1] - self.win_size + 1) // self.step )
            return np.float32(self.test[outter_index, inner_index:inner_index + self.win_size, :self.time_start_index]), np.float32(
                self.test_labels[outter_index, inner_index:inner_index + self.win_size]), np.float32(self.test[outter_index, inner_index:inner_index + self.win_size, self.time_start_index:])



        
class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        time_start_index = -5
        self.time_start_index = time_start_index
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]),np.float32(self.train[index:index + self.win_size, self.time_start_index:])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size]),np.float32(self.val[index:index + self.win_size, self.time_start_index:])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size]),np.float32(self.test[index:index + self.win_size, self.time_start_index:])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]),np.float32(self.test[index:index + self.win_size, self.time_start_index:])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


