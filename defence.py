import torch
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
import numpy as np
from copy import deepcopy
import random

import math
import matplotlib.pyplot as plt
class temp_args:
    num_agents = 10

args = temp_args()
#args = args_parser()
#########################improved Flame###############################
def improved_flame(grad_in, cluster_sel=0):    #adjusted cosine distance filter
    """The HDBSCAN filter based on cosine distance
    Args:
        grad_in (list/np.ndarray): the raw input weight_diffs
    """
    # distance_matrix = pairwise_distances(grad_in - grad_in.mean(axis=0), metric='cosine')   #adjusted cosine distance
    distance_matrix = pairwise_distances(grad_in, metric='cosine')
    return improved_flame_filter(distance_matrix, cluster_sel=cluster_sel)

def improved_flame_filter(inputs, cluster_sel=0):
    cluster_base = hdbscan.HDBSCAN(
        metric='l2',
        min_cluster_size=args.num_corrupt,  # the smallest size grouping that you wish to consider a cluster
        # approx_min_span_tree=True,
        # gen_min_span_tree=True,
        allow_single_cluster=False,  # False performs better in terms of Backdoor Attack
        min_samples=1,  # how conservative you want you clustering to be
    #    cluster_selection_epsilon=0,
    )
    cluster_lastLayer = hdbscan.HDBSCAN(
        metric='l2',
        min_cluster_size=2,
        allow_single_cluster=True,
        # approx_min_span_tree=True,
        # gen_min_span_tree=True,
        min_samples=1,
    )
    if cluster_sel == 0:
        cluster = cluster_base
    elif cluster_sel == 1:
        cluster = cluster_lastLayer
    cluster.fit(inputs)
    # cluster.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    # plt.show()
    label = cluster.labels_
    print("label: ",label)
    if (label == -1).all():
        bengin_id = np.arange(len(inputs)).tolist()
    else:
        label_class, label_count = np.unique(label, return_counts=True)
        if -1 in label_class:
            label_class, label_count = label_class[1:], label_count[1:]
        majority = label_class[np.argmax(label_count)]
        bengin_id = np.where(label == majority)[0].tolist()

    return bengin_id

#########################cosine_distance_filter#######################
def cosine_distance_filter(grad_in, cluster_sel=0):
    """The HDBSCAN filter based on cosine distance
    Args:
        grad_in (list/np.ndarray): the raw input weight_diffs
    """
    #distance_matrix = pairwise_distances(grad_in - grad_in.mean(axis=0), metric='cosine')  # adjusted cosine distance
    distance_matrix = pairwise_distances(grad_in, metric='cosine')
    return hdbscan_filter(distance_matrix, cluster_sel=cluster_sel)
###################################Flame##############################

def flame(grad_in, cluster_sel=0):
    """The HDBSCAN filter based on cosine distance
    Args:
        grad_in (list/np.ndarray): the raw input weight_diffs
    """
    # distance_matrix = pairwise_distances(grad_in - grad_in.mean(axis=0), metric='cosine')   #adjusted cosine distance
    distance_matrix = pairwise_distances(grad_in, metric='cosine')
    #print(distance_matrix)
    return flame_filter(distance_matrix, cluster_sel=cluster_sel)

def flame_filter(inputs, cluster_sel=0):
    cluster_base = hdbscan.HDBSCAN(
        #metric='l2',
        metric = 'precomputed',
        min_cluster_size=int(args.num_agents/2 + 1),  # the smallest size grouping that you wish to consider a cluster
        # approx_min_span_tree=True,
        # gen_min_span_tree=True,
        allow_single_cluster=True,  # False performs better in terms of Backdoor Attack
        min_samples=1,  # how conservative you want you clustering to be
    #    cluster_selection_epsilon=0,
    )
    cluster_lastLayer = hdbscan.HDBSCAN(
        metric='l2',
        min_cluster_size=2,
        allow_single_cluster=True,
        # approx_min_span_tree=True,
        # gen_min_span_tree=True,
        min_samples=1,
    )
    if cluster_sel == 0:
        cluster = cluster_base
    elif cluster_sel == 1:
        cluster = cluster_lastLayer
    cluster.fit(inputs)
    # cluster.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    # plt.show()
    label = cluster.labels_
    print("label: ",label)
    if (label == -1).all():
        bengin_id = np.arange(len(inputs)).tolist()
    else:
        label_class, label_count = np.unique(label, return_counts=True)
        if -1 in label_class:
            label_class, label_count = label_class[1:], label_count[1:]
        majority = label_class[np.argmax(label_count)]
        bengin_id = np.where(label == majority)[0].tolist()
    # FOR = np.size(np.where(label[0:args.num_corrupt] == 0))
    # FDR = np.size(np.where(label[args.num_corrupt:args.num_agents] == -1))

    return bengin_id

def hdbscan_filter(inputs, cluster_sel=0):
    cluster_base = hdbscan.HDBSCAN(
        metric='l2',
        min_cluster_size=2,  # the smallest size grouping that you wish to consider a cluster
        allow_single_cluster=True,  # False performs better in terms of Backdoor Attack
        min_samples=1,  # how conservative you want you clustering to be
        # cluster_selection_epsilon=0.1,
    )
    cluster_lastLayer = hdbscan.HDBSCAN(
        metric='l2',
        min_cluster_size=2,
        allow_single_cluster=True,
        min_samples=1,
    )
    if cluster_sel == 0:
        cluster = cluster_base
    elif cluster_sel == 1:
        cluster = cluster_lastLayer
    cluster.fit(inputs)
    label = cluster.labels_
    print("label: ",label)
    #if (label == -1).all():
    #    bengin_id = np.arange(len(inputs)).tolist()
    #else:
    #    label_class, label_count = np.unique(label, return_counts=True)
    #    if -1 in label_class:
    #        label_class, label_count = label_class[1:], label_count[1:]
    #    majority = label_class[np.argmax(label_count)]
    #    bengin_id = np.where(label == majority)[0].tolist()

    #return bengin_id
    return label
####################neups & diffs######################
def neups_filter(inputs, cluster_sel=0):
    cluster_base = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=2,  # the smallest size grouping that you wish to consider a cluster
        allow_single_cluster=True,  # False performs better in terms of Backdoor Attack
        min_samples=1,  # how conservative you want you clustering to be
        # cluster_selection_epsilon=0.1,
    )
    cluster_lastLayer = hdbscan.HDBSCAN(
        metric='l2',
        min_cluster_size=2,
        allow_single_cluster=True,
        min_samples=1,
    )
    if cluster_sel == 0:
        cluster = cluster_base
    elif cluster_sel == 1:
        cluster = cluster_lastLayer
    cluster.fit(inputs)
    label = cluster.labels_
    print("label: ",label)
    #if (label == -1).all():
    #    bengin_id = np.arange(len(inputs)).tolist()
    #else:
    #    label_class, label_count = np.unique(label, return_counts=True)
    #    if -1 in label_class:
    #        label_class, label_count = label_class[1:], label_count[1:]
    #    majority = label_class[np.argmax(label_count)]
    #    bengin_id = np.where(label == majority)[0].tolist()

    #return bengin_id
    return label

#########################NEUP_calc#######################
def neups_metric(grad_in: np.ndarray):
    """NEUPs measures the magnitude changes of neurons in the last layer
    and use them to provide a rough estimation of the output labels for
    the training data of the individual client
    Args:
        grad_in (np.ndarray): the raw input weight_diffs
    Returns:
        [np.ndarray]: 2-dimession NormalizEd Energies UPdate for clients
    """
    if args.data == 'minist' or 'fmnist':
        weight_index = 1290
        bias_index = 10
    elif args.data == 'cifar10':
        weight_index = 2570
        bias_index = 10
    neups = []
    for client_grad in grad_in: #number of clients
        energy_weights = client_grad[-weight_index:-bias_index].reshape((bias_index, -1)) #row = bias_index(num_neuron)
        energy_bias = client_grad[-bias_index::]
        energy_neuron = np.square(np.abs(energy_weights).sum(axis=1) + np.abs(energy_bias))
        energy_neuron /= energy_neuron.sum()
        neups.append(energy_neuron.tolist())

    return np.array(neups)

def neups_filter(grad_in): #grad_in: updates
    """The HDBSCAN filter based on NEUPS
    Args:
    grad_in (list/np.ndarray): the raw input weight_diffs
    """

    neups = neups_metric(grad_in)
    return hdbscan_filter(neups)

#########################TE(Threshold Exceedings)_calc#######################
def te_metric(neups):
    """TEs analyzes the parameter updates of the output layer for a model
    to measure the homogeneity of its training data
    Args:
        neups ([np.ndarray]): NormalizEd Energies UPdate
    Returns:
        [np.ndarray]: the number of threshold exceedings
    """
    #if args.data == 'minist' or 'fmnist' or 'cifar10':
    #bias_index = 10
    #elif args.data == 'tiny':
    #    bias_index = 10
    te = []
    bias_index = 10
    for client_neups in neups:
        threshold = max(0.01, 1 / bias_index) * client_neups.max()
        te.append((client_neups > threshold).sum())

    return te

#########################DDifs_calc#######################
def ddifs_metric(model, grad_in: np.ndarray, seed, samples_size=20000):
       # DDifs measures the difference of predicted scores between local update model
       # and global model as they provide information about distribution of the training
       # labels of the respective client
       # Args:
       #     grad_in (np.ndarray): the raw input weight_diffs
       #     samples_size (int, optional): the number of random samples. Defaults to 20000.
       # Returns:
       #     [list]: the DDifs for 3 different seeds as a list of 3 lists

    ddifs = []
    random.seed(seed)
    if args.data == 'mnist' or 'fmnist':
        random_samples = torch.randn(samples_size, 1, 28, 28, device = args.device)
    elif args.data =='cifar10':
        random_samples = torch.randn(samples_size, 3, 32, 32, device=args.device)
    # cifar10 tiny-imagenet else:
    for client_grad in grad_in:
        temp_model = deepcopy(model)  #global_model
        upgrade(client_grad.tolist(), temp_model)  #local_model
        temp_output = temp_model.forward(random_samples).detach().cpu().data
        model_output = model.forward(random_samples).detach().cpu().data
        neuron_diff = torch.div(temp_output, model_output).sum(axis=0) / samples_size
        ddifs.append(neuron_diff.numpy().tolist())

    return ddifs


def upgrade(grad_in: list, model):
    layer = 0
    _level_length = [0]
    for param in model.parameters():
        _level_length.append(param.data.numel() + _level_length[-1])
    for param in model.parameters():
        layer_diff = grad_in[_level_length[layer]:_level_length[layer + 1]]  #list[layer parameter numbers]
        param.data += torch.tensor(layer_diff, device=args.device).view(param.data.size())
        layer += 1

#########################Dpsight_cluster#######################
def dpsight_cluster(weights,model):
    cos_label = hdbscan_filter(dp_cos_dist(weights), cluster_sel=0)  ####dp_cos_dist calc bias dist
    cos_dist = (cos_label[:, None] == cos_label) * 1    #array 10 x 10
    neups_label = neups_filter(weights)
    neups_dist = (neups_label[:, None] == neups_label) * 1  #array
    ddifs_dist = np.zeros((args.num_agents, args.num_agents))
    for seed in range(3):
        ddifs_label = hdbscan_filter(ddifs_metric(model, weights, seed, samples_size=1))
        ddifs_dist += (np.array([x == y for x in ddifs_label for y in ddifs_label]) * 1).reshape(args.num_agents,args.num_agents)  # list
    merged_ddif_clust_dist = ddifs_dist / 3
    merged_dist = (merged_ddif_clust_dist + neups_dist + cos_dist) / 3
    clusters = hdbscan_filter(merged_dist)
    return clusters

#########################Dpsight_filter#######################
def dpsight_filter(weights,model):
    neups = neups_metric(weights)
    te = np.array(te_metric(neups))     #list: num of clients to array
    class_boundary = np.median(te) / 2
    label = np.empty(args.num_agents)
    for i in range(args.num_agents):
        label[i] = 1 if te[i] <= class_boundary else 0

    return label

#########################Dpsight_cos_filter#######################
def dp_cos_dist(weights):
    if args.data == 'minist' or 'fmnist':
        bias_index = 10
    elif args.data == 'cifar10':
        bias_index = 10
    cos_dist = pairwise_distances(weights[:,-bias_index:], metric='cosine')
    #a = weights[:,-bias_index:]
    return cos_dist

#########################Dpsight#######################
def dpsight(weights,model):
    accepted_models = []
    grad_in=weights.tolist()
    amount_of_positives = 0
    cluster = dpsight_cluster(weights, model)
    print("cluster: ",cluster)
    label = dpsight_filter(weights, model)
    print("label: ", label)
    label_, indices = np.unique(cluster, return_counts=True)
    for i in label_:
        a = np.squeeze(np.array(np.where(cluster==i)))
        print(a)
        label=np.array(label)
        amount_of_positives = label[a].sum()/indices[i]
        if amount_of_positives < 1/3 :

            if type(a.tolist())==int:
                accepted_models.append(grad_in[a.tolist()])
            else:
              for j in a.tolist():
                accepted_models.append(grad_in[j])
    print("len(accepted_models): ", len(accepted_models))

    return accepted_models

