from torch.utils.data import Dataset
from torchvision import datasets, transforms

import torch
import numpy as np
import random
import math
import os
import pickle
global_attack_mode = None
class cifar10_EC(Dataset):
  def __init__(self, father_set, **kwargs):
    self.dataset = father_set

  def __len__(self):
      return len(self.dataset)

  def __getitem__(self, idx):
      return  (self.dataset[idx], 2)

class femnist_EC(Dataset):
  def __init__(self, father_set, **kwargs):
    self.dataset = father_set

  def __len__(self):
      return len(self.dataset)

  def __getitem__(self, idx):
      return  (self.dataset[idx], 1)
  
class OwnCifar10(datasets.CIFAR10):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.non_iid_id = []
    self.test = False
  def __len__(self):
    if self.test == False:
      return len(self.non_iid_id)
    else:
      return super().__len__()

  def __getitem__(self, idx):
      if self.test == False:
        temp_list = list(super().__getitem__(self.non_iid_id[idx]))
      else:
        temp_list = list(super().__getitem__(idx))
      return tuple(temp_list)
      

class SubCifar10(Dataset):
  def __init__(self, father_set, **kwargs):
    self.non_iid_id = []
    self.father_set = father_set

  def __len__(self):
      return len(self.non_iid_id)

  def __getitem__(self, idx):
      return  self.father_set.__getitem__(self.non_iid_id[idx])
  

class General_Dataset(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, data, targets, users_index = None, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform
        if users_index != None:
            self.users_index = users_index
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform != None:
            img = self.transform(img)
        return (img, label)

class SubTiny(Dataset):
  def __init__(self, father_set, **kwargs):
    self.non_iid_id = []
    self.father_set = father_set

  def __len__(self):
      return len(self.non_iid_id)

  def __getitem__(self, idx):
      return  self.father_set.__getitem__(self.non_iid_id[idx])
      
def load_imagenet(path, transform = None):
    imagenet_list = torch.load(path)
    data_list = []
    targets_list = []
    for item in imagenet_list:
        data_list.append(item[0])
        targets_list.append(item[1])
    targets = torch.LongTensor(targets_list)
    return General_Dataset(data = data_list, targets=targets, transform=transform)

  
class Fede_Dataset(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, data, targets, users_index = None, transform = None):
        self.data = data
        self.targets = targets
        self.transform = transform
        if users_index != None:
            self.users_index = users_index
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform != None:
            img = self.transform(img)
        return img, label 
    
def load_femnist(path, train = True, transform = None):
    femnist_dict = None
    if train == True:
        with open(path, "rb") as f:
            femnist_dict = pickle.load(f)
    else:
        femnist_dict = torch.load(path)
    
    training_data = femnist_dict['training_data']
    targets = femnist_dict['targets']
    user_idx = femnist_dict['user_idx']

    for i in range(len(training_data)):
        training_data[i] = torch.tensor(training_data[i].reshape(1,28,28)).float()

    targets = torch.LongTensor(targets)
    return Fede_Dataset(data = training_data, targets=targets, users_index = user_idx, transform=transform)

class SubFedeMnist(Dataset):
  def __init__(self, id, father_set, **kwargs):
    self.id = id
    self.father_set = father_set

  def __len__(self):
      return len(self.id)


  def __getitem__(self, index):
      temp_list = list(self.father_set.__getitem__(self.id[index]))
      return tuple(temp_list)
  
def load_dataset(dataset_name, path):
   if dataset_name == 'cifar10':
        transforms_list = []
        transforms_list.append(transforms.ToTensor())

        if global_attack_mode == 'edge_case':
            transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        mnist_transform = transforms.Compose(transforms_list)
        train_dataset = OwnCifar10(root = path, train=True, download=True, transform=mnist_transform)
        test_dataset = OwnCifar10(root = path, train=False, download=True, transform=mnist_transform)
        train_dataset.test = True
        test_dataset.test = True
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)
        return train_dataset, test_dataset
   
   elif dataset_name == 'tiny':
        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        mnist_transform = transforms.Compose(transforms_list)

        train_dataset = load_imagenet(os.path.join(path, 'tiny-imagenet-pt', 'imagenet_train.pt'), transform=mnist_transform)
        test_dataset = load_imagenet(os.path.join(path, 'tiny-imagenet-pt', 'imagenet_val.pt'), transform=mnist_transform)
        
        return train_dataset, test_dataset
   elif dataset_name == 'femnist':
        transforms_list = []
        
        if global_attack_mode == 'edge_case':
            transforms_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        train_dataset = load_femnist(os.path.join(path, 'femnist_training.pickle'), train = True, transform = None)
        test_dataset = load_femnist(os.path.join(path, 'femnist_test.pt'), train = False, transform = None)
        return train_dataset, test_dataset
   
   elif dataset_name == 'fashionmnist':
        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        mnist_transform = transforms.Compose(transforms_list)
        train_dataset = datasets.FashionMNIST(root = path, train=True, download=True, transform=mnist_transform)
        test_dataset = datasets.FashionMNIST(root = path, train=False, download=True, transform=mnist_transform)
        return train_dataset, test_dataset
   
def distribution_data_dirchlet(dataset, n_classes = 10, num_of_agent = 10):
        if num_of_agent == 1:
            return {0:range(len(dataset))}
        N = dataset.targets.shape[0]
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(num_of_agent)]
        for k in range(n_classes):
            idx_k = np.where(dataset.targets == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(0.5, num_of_agent))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]


        for j in range(num_of_agent):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        return net_dataidx_map
def synthetic_real_word_distribution(dataset, num_agents):
        num_user = len(dataset.users_index)
        u_train = dataset.users_index
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, num_agents)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(num_agents)}

        for i in range(num_agents):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))

        return net_dataidx_map

def split_femnist(train_dataset, num_of_agent):
      net_dataidx_map = synthetic_real_word_distribution(train_dataset, num_of_agent)
      random.shuffle(net_dataidx_map)
      boring_list = []

      train_loader_list = []
      for index in range(num_of_agent):
        tempSet = SubFedeMnist(id = net_dataidx_map[index], father_set = train_dataset)
        boring_list.append(tempSet)
        train_loader_list.append(torch.utils.data.DataLoader(tempSet, batch_size = 64, shuffle = True))
      return train_loader_list

def split_train_data(train_dataset, num_of_agent = 10, non_iid = False, n_classes = 10):
    if non_iid == False:
        average_num_of_agent = math.floor(len(train_dataset) / num_of_agent)
        train_dataset_list = torch.utils.data.random_split(train_dataset, [average_num_of_agent] * num_of_agent)
        random.shuffle(train_dataset_list)
        train_loader_list = []
        for index in range(num_of_agent):
            train_loader_list.append(torch.utils.data.DataLoader(train_dataset_list[index], batch_size = 256, shuffle = True))
    else:
        net_dataidx_map = distribution_data_dirchlet(train_dataset, n_classes = n_classes, num_of_agent = num_of_agent)
        train_loader_list = []
        for index in range(num_of_agent):
            temp_train_dataset = SubCifar10(train_dataset)
            temp_train_dataset.non_iid_id = net_dataidx_map[index]
            train_loader_list.append(torch.utils.data.DataLoader(temp_train_dataset, batch_size = 256, shuffle = True))
    return train_loader_list
