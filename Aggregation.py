
import torchvision
import torch
from classifier_models.resnet_cifar import ResNet18
from torchsummary import summary
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import math
import copy
import numpy as np
import sklearn.metrics.pairwise as smp
import wandb
from defence import *

device = None
num_of_agent = None
using_wandb = None
num_of_malicious = None
lr = 1

def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)



def pairwise_distance(w_locals):
    
    vectors = w_locals
    distance = torch.zeros([len(vectors), len(vectors)])
    
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                
    return distance

def sparse_fed_topk(vec, k):
    topkVals = torch.zeros(k).to(device = device)
    topkIndices = torch.zeros(k).long().to(device = device)
    torch.topk(vec**2, k, sorted=False, out=(topkVals, topkIndices))

    ret = torch.zeros_like(vec).to(device = device)
    ret[topkIndices] = vec[topkIndices].to(device = device)
    return ret

def weighted_average_oracle(agent_updates_dict,weight):
    sm_updates = 0
    total_sum = sum(weight)
    for _id, update in agent_updates_dict.items():
        sm_updates += update * (weight[_id] / total_sum)
    return  sm_updates

def l2dist(p1, p2):
    squared_sum = 0
    squared_sum = torch.sum(torch.pow(p1- p2, 2))
    return math.sqrt(squared_sum)


def init_sparsefed(model):
    global Vvelocity
    global Verror
    shape = (len(parameters_to_vector(model.parameters())),)
    Vvelocity = torch.zeros(shape).to(device = device)
    Verror = torch.zeros(shape).to(device = device)

def init_foolsgold(model):
    global foolsgold_memory
    shape = len(parameters_to_vector(model.parameters()))
    foolsgold_memory = np.zeros((num_of_agent, shape))

def extra_analysis(aggregation_dict):
    normal_trigger_update = [aggregation_dict[0]]
    gene_trigger_update = [aggregation_dict[1]]
    topk_trigger_update = [aggregation_dict[2]]
    normal_update = []
    for i in range(3, 50):
        normal_update.append(aggregation_dict[i])
    
    normal_trigger_cos = num_dif_of_two_list(normal_trigger_update, normal_update, 'cos')
    normal_gene_trigger_cos = num_dif_of_two_list(gene_trigger_update, normal_update, 'cos')
    topk_trigger_cos = num_dif_of_two_list(topk_trigger_update, normal_update, 'cos')
    norm_norm = num_dif_of_one_list(normal_update, 'cos')
    print('normal  cos')
    print(normal_trigger_cos)
    print('normal trigger cos')
    print(normal_gene_trigger_cos)
    print('topk trigger cos')
    print(topk_trigger_cos)
    print('normal vs normal update cos')
    print(norm_norm)
    if using_wandb:
        wandb.log({"old_trigger": normal_trigger_cos, "trigger_generation":normal_gene_trigger_cos, "topk_trigger_generation":topk_trigger_cos, "normal":norm_norm})

def single_analysis(aggregation_dict):
    trigger_update = []
    for i in range(1):
        trigger_update.append(aggregation_dict[i])
    normal_update = []
    for i in range(1, 10):
        normal_update.append(aggregation_dict[i])
    
    trigger_normal = num_dif_of_two_list(trigger_update, normal_update, 'cos')
    normal_normal = num_dif_of_one_list(normal_update, 'cos')
    trigger_trigger = num_dif_of_one_list(trigger_update, 'cos')
    print('trigger vs normal cos')
    print(trigger_normal)
    print('normal vs normal update cos')
    print(normal_normal)
    print('trigger vs trigger update cos')
    print(trigger_trigger)
    if using_wandb:
        wandb.log({"trigger_trigger": trigger_trigger, "trigger_normal": trigger_normal,  "normal_normal":normal_normal})

def benign_analysis(aggregation_dict):
    normal_update = []
    for i in range(0, 50):
        normal_update.append(aggregation_dict[i])
    normal_normal = num_dif_of_one_list(normal_update, 'cos')
    if using_wandb:
        wandb.log({"normal_normal":normal_normal})

def aggregation_time(model, agent_updates_dict, clip = 0, underwater = False, agg_way = None, random_list = None):
        def clip_updates(agent_updates_dict, clip):
            for update in agent_updates_dict.values():
                l2_update = torch.norm(update, p=2) 
                update.div_(max(1, l2_update/clip))
            return
            
        def agg_flame(agent_updates_dict):
          """ fed avg with flame """
          update_len = len(agent_updates_dict.keys())
          weights = np.zeros((update_len, np.array(len(agent_updates_dict[0]))))
          for _id, update in agent_updates_dict.items():
              weights[_id] = update.cpu().detach().numpy()  # np.array
          # grad_in = weights.tolist()  #list
          benign_id = flame(weights, cluster_sel=0)
          print('!!!FLAME: remained ids are:')
          print(benign_id)
          accepted_models_dict = {}
          escaped_num = 0
          for i in range(len(benign_id)):
              if benign_id[i] < num_of_malicious:
                escaped_num += 1
              accepted_models_dict[i] = torch.tensor(weights[benign_id[i], :]).to(device)
          sm_updates, total_data = 0, 0
          for _id, update in accepted_models_dict.items():
              n_agent_data = 1
              sm_updates += n_agent_data * update
              total_data += n_agent_data
          if using_wandb:
            wandb.log({"escaped_num": escaped_num})
          return sm_updates / total_data, benign_id

        def agg_avg(agent_updates_dict, underwater = False, random_list = None):
            """ classic fed avg """
            sm_updates, total_data = 0, 0
            for _id, update in agent_updates_dict.items():
                if underwater == True and _id < num_of_malicious:
                  continue
                if random_list != None and _id not in random_list:
                  continue
                sm_updates +=   update
                total_data += 1 
            return  sm_updates / total_data
        
        def agg_comed(agent_updates_dict):
          agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
          concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
          return torch.median(concat_col_vectors, dim=1).values
        
        def agg_sign(agent_updates_dict):
          """ aggregated majority sign update """
          agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
          sm_signs = torch.sign(sum(agent_updates_sign))
          return torch.sign(sm_signs)
        
        def multi_krum(agent_updates_dict):
          selected_number = 5
          tolerance_number = 0
          update_len = len(agent_updates_dict.keys())
          #aggregation method is averaging in this case
          if selected_number >= update_len:
              return agg_avg(agent_updates_dict)
          else:
              # Compute list of scores
              scores = [list() for i in range(update_len)]
              for i in range(update_len - 1):
                  score = scores[i]
                  for j in range(i + 1, update_len):
                      # With: 0 <= i < j < nbworkers
                      distance = torch.dist(agent_updates_dict[i], agent_updates_dict[j]).item()
                      #if distance == float('nan'):
                          #distance = float('inf')
                      score.append(distance)
                      scores[j].append(distance)
              nbinscore = update_len - tolerance_number - 2
              for i in range(update_len):
                  score = scores[i]
                  score.sort()
                  scores[i] = sum(score[:nbinscore])
              # Return the average of the m gradients with the smallest score
              pairs = [(agent_updates_dict[i], scores[i]) for i in range(update_len)]
              pairs.sort(key=lambda pair: pair[1])
              result = pairs[0][0]
              for i in range(1, selected_number):
                  result = result + pairs[i][0]
              result = result / float(selected_number)
              #print(result)
              return result
          
        def trimmed_mean(agent_updates_dict):
          c = num_of_malicious
          n = len(agent_updates_dict) - 2 * c
          update_list = []
          for _id, update in agent_updates_dict.items():
            update_list.append(update)
          distance = pairwise_distance(update_list)
          distance = distance.sum(dim=1)

          med = distance.median()
          _, chosen = torch.sort(abs(distance - med))
          chosen = chosen[: n]
          result = update_list[chosen[0]]

          for i in range(1, n):
              result += update_list[chosen[i]]
          result = result / n
          return result
        
        #0.001 for sign
        server_lr = lr
        n_params = len(agent_updates_dict[0])
        
        lr_vector = torch.Tensor([server_lr] * n_params).to(device = device)

        if clip != 0:
            clip_updates(agent_updates_dict, clip)

        def sparse_fed(agent_updates_dict):
            aggregated_updates = agg_avg(agent_updates_dict)
            global Vvelocity
            global Verror
            rho = 0.9
            torch.add(aggregated_updates,
                      Vvelocity,
                      alpha=rho,
                      out=Vvelocity)
            Verror += Vvelocity
            update = sparse_fed_topk(Verror, k = 1500000)

            Verror[update.nonzero()] = 0
            Vvelocity[update.nonzero()] = 0
            return update

        def foolsgold(agents_updates_dict):
          global foolsgold_memory
          print(foolsgold_memory)
          for i in range(len(agents_updates_dict)):
            foolsgold_memory[i] += agents_updates_dict[i].detach().cpu().numpy()

          n_clients = len(agent_updates_dict)
          cs = smp.cosine_similarity(foolsgold_memory) - np.eye(n_clients)
          maxcs = np.max(cs, axis=1)
          # pardoning
          for i in range(n_clients):
              for j in range(n_clients):
                  if i == j:
                      continue
                  if maxcs[i] < maxcs[j]:
                      cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
          wv = 1 - (np.max(cs, axis=1))
          wv[wv > 1] = 1
          wv[wv < 0] = 0
          alpha = np.max(cs, axis=1)
          # Rescale so that max value is wv
          wv = wv / np.max(wv)
          wv[(wv == 1)] = .99
          # Logit function
          wv = (np.log(wv / (1 - wv)) + 0.5)
          wv[(np.isinf(wv) + wv > 1)] = 1
          wv[(wv < 0)] = 0
          sm_updates, total_data = 0, 0
          print('wv = ')
          print(wv)
          for _id, update in agent_updates_dict.items():
              sm_updates += wv[_id] * update
              total_data += 1 
          return  sm_updates / total_data
        
        def RFA(agents_updates_dict):
            num_of_client = len(agent_updates_dict)
            alphas = [1] * num_of_client
            median = weighted_average_oracle(agents_updates_dict, weight = alphas)
            maxiter = 10
            eps=1e-5
            update_list = []
            for i in range(num_of_client):
               update_list.append(agents_updates_dict[i])

            for i in range(maxiter):
              prev_median = median
              weights = [alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alphas, update_list)]
              weights = [i / sum(weights) for i in weights]
              print('weights here')
              print(weights)
              median =  weighted_average_oracle(agents_updates_dict, weight = weights)
            return median

        if agg_way == 'flame':
          aggregated_updates, benign_id = agg_flame(agent_updates_dict)
        elif agg_way == 'avg':
          aggregated_updates = agg_avg(agent_updates_dict, underwater = underwater, random_list = random_list)
        elif agg_way == 'median':
           aggregated_updates = agg_comed(agent_updates_dict)
        elif agg_way == 'sign':
           aggregated_updates = agg_sign(agent_updates_dict)
        elif agg_way == 'krum':
           aggregated_updates = multi_krum(agent_updates_dict)
        elif agg_way == 'trimmed_mean':
           aggregated_updates = trimmed_mean(agent_updates_dict)
        elif agg_way == 'sparsefed':
           aggregated_updates = sparse_fed(agent_updates_dict)
        elif agg_way == 'foolsgold':
           aggregated_updates = foolsgold(agent_updates_dict)
        elif agg_way == 'RFA':
           aggregated_updates = RFA(agent_updates_dict)
        else:
          print('unknown aggregation')

        cur_global_params = parameters_to_vector(model.parameters())
        new_global_params =  (cur_global_params + lr_vector * aggregated_updates).float() 
        vector_to_parameters(new_global_params, model.parameters())
        if agg_way == 'flame':
          return benign_id
def aggregate_batch_norm(model, agent_updates_dict, random_list = None):
    update_state_dict = copy.deepcopy(agent_updates_dict[0])
    for name, data in update_state_dict.items():
          update_state_dict[name] = torch.zeros_like(data)

    for name, _ in agent_updates_dict[0].items():
        count = 0
        for _id, _ in agent_updates_dict.items():
            if random_list != None and _id not in random_list:
                continue
            count += 1
            update_state_dict[name] = update_state_dict[name] + agent_updates_dict[_id][name]

        update_state_dict[name] = update_state_dict[name] / count
    
    model_state_dict = model.state_dict()

    for name, _ in update_state_dict.items():
        model_state_dict[name] = model_state_dict[name] + update_state_dict[name]

    model.load_state_dict(model_state_dict)



def num_dif_of_two_list(list_1, list_2, criterion):
  total_distance = 0
  count = 0
  for i in range(len(list_1)):
    for j in range(len(list_2)):
      vector_i = list_1[i]
      vector_j = list_2[j]
      if criterion == 'l2':
          total_distance += torch.dist(vector_i, vector_j).item()
      elif criterion == 'cos':
        
        cos = nn.CosineSimilarity(dim=0)
        #print(cos(vector_i, vector_j))
        total_distance += cos(vector_i, vector_j).item()
      count += 1
  return total_distance / count
  
def num_dif_of_one_list(list_1, criterion):
  total_distance = 0
  count = 0
  for i in range(len(list_1)):
    for j in range(i + 1, len(list_1)):
      vector_i = list_1[i]
      vector_j = list_1[j]
      if criterion == 'l2':
          total_distance += torch.dist(vector_i, vector_j).item()
      elif criterion == 'cos':
        cos = nn.CosineSimilarity(dim=0)
        #print(cos(vector_i, vector_j))
        total_distance += cos(vector_i, vector_j).item()
      count += 1
  if count == 0:
    return 0
  return total_distance / count
  
def get_average_norm(aggregation_dict):
  sum = 0
  count = 0
  for _id, update in aggregation_dict.items():
    if _id >= num_of_malicious:
      count += 1
      sum += torch.norm(update, p = 2).item()
  
  return sum/count


def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0+size*i : size*(i+1)] for i in range(n)]

    leftover = ylen - size*n
    edge = size*n
    for i in range(leftover):
            chunks[i%n].append(ys[edge+i])

    return chunks
    
def layer_equal_division(model, single_equal_division):

    parameter_distribution = []
    total_division = []
    raw_divided_part = []

    for para in model.parameters():
        size = para.view(-1).shape[0]
        parameter_distribution.append(size)
    
    count = 0
    for layer_size in parameter_distribution:
        temp_set = set(range(layer_size))
        temp_chunk_list = chunk(range(layer_size), single_equal_division)
        
        copied_temp_chunk_list = copy.deepcopy(temp_chunk_list)
        for agent_index in range(len(copied_temp_chunk_list)):
            for para_index in range(len(copied_temp_chunk_list[agent_index])):
                copied_temp_chunk_list[agent_index][para_index] = copied_temp_chunk_list[agent_index][para_index] + count

        raw_divided_part.append(copy.deepcopy(copied_temp_chunk_list))
        for index in range(len(temp_chunk_list)):
            temp_chunk_list[index] = list(temp_set - set(temp_chunk_list[index]))

        total_division.append(temp_chunk_list)
        count += layer_size
    
    final_divided_part = []
    for index in range(single_equal_division):
        final_divided_part.append([])

    for layer_index in range(len(raw_divided_part)):
        for agent_index in range(len(raw_divided_part[layer_index])):
            final_divided_part[agent_index].extend(raw_divided_part[layer_index][agent_index]) 

    return total_division, final_divided_part

def get_batch_norm_list(input_model):
  first_set = set()
  for name, data in input_model.state_dict().items():
      first_set.add(name)


  second_set = set()
  for name, W in input_model.named_parameters():
      second_set.add(name)
  
  return list(first_set - second_set)


def initialize_batch_norm_list(input_model, batch_norm_list):
    agent_list = []
    for i in range(num_of_agent + 1):
        agent_list.append(dict())
    
    model_state_dict = input_model.state_dict()

    for i in range(num_of_agent + 1):
      for name in batch_norm_list:
          agent_list[i][name] = copy.deepcopy(model_state_dict[name].detach())

    return agent_list

        

def load_batch_norm(temp_model, agent_id, batch_norm_list, agent_batch_norm_list):
    temp_agent_dict = agent_batch_norm_list[agent_id]
    model_dict = temp_model.state_dict()

    for name in batch_norm_list:
        #print(name)
        model_dict[name].copy_(copy.deepcopy(temp_agent_dict[name].detach()))

def save_batch_norm(temp_model, agent_id, batch_norm_list, agent_batch_norm_list):
      temp_agent_dict = agent_batch_norm_list[agent_id]
      model_dict = temp_model.state_dict()

      for name in batch_norm_list:
          temp_agent_dict[name] = copy.deepcopy(model_dict[name].detach())


def copy_params(model, state_dict, coefficient_transfer=100):

    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            shape = param.shape
            #random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(torch.cuda.FloatTensor)
            # negative_tensor = (random_tensor*-1)+1
            # own_state[name].copy_(param)
            own_state[name].copy_(param.clone())