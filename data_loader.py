from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import numpy as np
import random
import math

class OwnCifar10(datasets.CIFAR10):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.heatmap = []
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
      temp_list.append(0)
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
        self.heatmap = []
        if users_index != None:
            self.users_index = users_index
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform != None:
            img = self.transform(img)
        if len(self.heatmap) == len(self):
            return (img, label, self.heatmap[item])
        else:
            return (img, label, 0)

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



def load_dataset(dataset_name, path):
   if dataset_name == 'cifar10':
        transforms_list = []
        transforms_list.append(transforms.ToTensor())

        mnist_transform = transforms.Compose(transforms_list)
        train_dataset = OwnCifar10(root = '../data', train=True, download=True, transform=mnist_transform)
        test_dataset = OwnCifar10(root = '../data', train=False, download=True, transform=mnist_transform)
        train_dataset.test = True
        test_dataset.test = True
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)
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
