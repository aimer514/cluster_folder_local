from AutoEncoder import *
import torch
import copy
import numpy as np

def test_model(model, test_loader):
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device = auto_device)
        target = target.to(device = auto_device)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()
    acc = correctly_labeled_samples / total_test_number
    print('benign accuracy  = {}'.format(acc))
    return acc

def clip_image(x):
  return torch.clamp(x, 0, 1.0)

def poison_data_only_target(data, target, target_label):
  data = copy.deepcopy(data)
  target = copy.deepcopy(target)
  target_tensor = []
  for index in range(len(target)):
          target[index] = target_label
  random_perm = torch.randperm(len(data))
  data = data[random_perm]
  target = target[random_perm]
  return data.to(device = auto_device),target.to(device = auto_device)

def poison_data_add_noise(data, target, target_label, noise_model  = None, norm_bound = 6.5, poison_frac = 0.2):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)

    target_tensor = []
    poison_number = math.floor(len(target) * poison_frac)

    produced_noise = noise_model(data.to(device = auto_device)).detach()
    for index in range(poison_number):
            target[index] = target_label

    for tensor_index in range(len(produced_noise)):
      norm_cut = max(1, torch.norm(produced_noise[tensor_index], p=2) / norm_bound)
      produced_noise[tensor_index] = produced_noise[tensor_index] / norm_cut

    data[0:poison_number] = clip_image(data[0:poison_number].to(device = auto_device) + produced_noise[0:poison_number].to(device = auto_device))


    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    target = target[random_perm]

    return data.to(device = auto_device), target.to(device = auto_device)
    
def test_mali_noise(model, noise_model, test_loader, target_label, norm_bound = 6.5):
    noise_model.eval()
    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = poison_data_only_target(data, target, target_label)

        noise = noise_model(data)
        norm_cut = max(1, torch.norm(noise, p=2) / (norm_bound * math.floor(math.sqrt(test_loader.batch_size))))
        noise = noise / norm_cut
        #print(torch.norm(noise, p = 2))
        current_data = clip_image(data + noise)

        output = model(current_data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        #print('pred_labels is ')
        #print(pred_labels)
        #print('target is')
        #print(target)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    noise_model.train()
    print('mali accuracy  = {}'.format(acc))
    return acc

def train_noise_model(classification_model, target_label, agent_train_loader, norm_for_one_sample, input_noise_model = None):
    classification_model.eval()
    if input_noise_model == None:
      noise_model  = Autoencoder().to(device = auto_device).to(device = auto_device)
    else:
      noise_model = input_noise_model
      
    noise_model.train()
    noise_optimizer = torch.optim.Adam(noise_model.parameters(), lr = 0.01)
    final_model = None
    best_acc = 0
    backdoor_epoch_num = 30

    for epoch in range(backdoor_epoch_num):
        temp_count = 0
        for batch_idx, (data, target) in enumerate(agent_train_loader):

            noise_optimizer.zero_grad()
            data, target = poison_data_only_target(data, target, target_label)

            noise = noise_model(data)
            if temp_count % 50 == 0:
              pass
              #print(torch.norm(noise, p=2))
            norm_cut = max(1, torch.norm(noise, p=2) / (norm_for_one_sample * math.floor(math.sqrt(agent_train_loader.batch_size))))
            
            noise = noise / norm_cut
            data = clip_image(data + noise)

            output = classification_model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()
            noise_optimizer.step()

            temp_count += 1
            if temp_count % 500 == 0:
              print(loss)

    classification_model.train()
    return noise_model

def train_mali_model_with_noise(classification_model, noise_model, target_label, agent_train_loader, norm_for_one_sample):

    training_epoch = 5
    noise_model.eval()
    classification_model.train()
    poison_frac = 0.2

    mali_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01, )
    for epoch in range(training_epoch):
        total_loss = 0
        temp_count = 0
        
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            mali_optimizer.zero_grad()

            data, target = poison_data_add_noise(data, target, target_label, noise_model  = noise_model, norm_bound = norm_for_one_sample, poison_frac = poison_frac)

            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()

            mali_optimizer.step()

            temp_count += 1
            if temp_count % 500 == 0:
              print(loss)

    noise_model.train()



def train_benign_model(classification_model, agent_train_loader):
    #5
    training_epoch = 5
    classification_model.train()
    benign_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.001, )
    for epoch in range(training_epoch):
        temp_count = 0
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            data = data.to(device = auto_device)
            target = target.to(device = auto_device)
            benign_optimizer.zero_grad()
            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()
            benign_optimizer.step()

            temp_count += 1
            if temp_count % 500 == 0:
              print(loss)
        #print('benign accuracy for benign model is')
        #test_model(classification_model, test_loader)

def get_topk(model, mali_update, topk_ratio = 0.2):
    mali_layer_list = []
    parameter_distribution = [0]
    total = 0

    for para in model.parameters():
        size = para.view(-1).shape[0]
        total += size
        parameter_distribution.append(total)
    
    _, indices = torch.topk(mali_update.abs(), math.floor(len(mali_update) * topk_ratio), largest = False)
    mask_flat_all_layer = torch.zeros(len(mali_update)).cuda()
    mask_flat_all_layer[indices] = 1.0

    count = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            mask_flat = mask_flat_all_layer[count:count + gradients_length]
            mali_layer_list.append(mask_flat.reshape(parms.size()).cuda())
            count += gradients_length
            
    return mali_layer_list

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
        layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

def poison_data_with_normal_trigger(data, target, target_label, poison_frac = 0.2, agent_no = -1):
    data = copy.deepcopy(data)
    target = copy.deepcopy(target)
    
    target_tensor = []
    poison_number = math.floor(len(target) * poison_frac)
    trigger_value = 0
    pattern_type = [[[0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 6], [0, 7], [0, 8], [0, 9]],
    [[3, 0], [3, 1], [3, 2], [3, 3]],
    [[3, 6], [3, 7], [3, 8], [3, 9]]]
    if agent_no == -1:
        for index in range(poison_number):
                target[index] = target_label
                for channel in range(3):
                  for i in range(len(pattern_type)):
                      for j in range(len(pattern_type[i])):
                          pos = pattern_type[i][j]
                          data[index][channel][pos[0]][pos[1]] = trigger_value
    else:
        for index in range(poison_number):
            target[index] = target_label
            for channel in range(3):
              for j in range(len(pattern_type[agent_no])):
                  pos = pattern_type[agent_no][j]
                  data[index][channel][pos[0]][pos[1]] = trigger_value



    random_perm = torch.randperm(len(data))
    data = data[random_perm]
    target = target[random_perm]

    return data.to(device = auto_device), target.to(device = auto_device)

def test_mali_normal_trigger(model, test_loader, target_label):


    total_test_number = 0
    correctly_labeled_samples = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = poison_data_with_normal_trigger(data, target, target_label, poison_frac = 1.0)
        output = model(data)
        total_test_number += len(output)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        #print('pred_labels is ')
        #print(pred_labels)
        #print('target is')
        #print(target)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, target)).item()
    model.train()

    acc = correctly_labeled_samples / total_test_number
    print('mali accuracy  = {}'.format(acc))
    return acc

def train_mali_model_with_normal_trigger(classification_model, target_label, agent_train_loader, agent_no = -1):
    classification_model.train()
    training_epoch = 5


    mali_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01, )
    for epoch in range(training_epoch):
        total_loss = 0
        temp_count = 0
        
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            mali_optimizer.zero_grad()
            #0.05 for vgg, 0.2 for resnet
            data, target = poison_data_with_normal_trigger(data, target, target_label, poison_frac = 0.2, agent_no = agent_no)

            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))
            
            loss.backward()

            mali_optimizer.step()

            temp_count += 1
            if temp_count % 500 == 0:
              print(loss)

def train_mali_model_with_normal_trigger_topk_mode(classification_model, target_label, agent_train_loader):
    initial_global_model_params = parameters_to_vector(classification_model.parameters()).detach()
    classification_model.train()
    train_benign_model(classification_model, agent_train_loader)

    with torch.no_grad():
      mali_update = parameters_to_vector(classification_model.parameters()).double() - initial_global_model_params

    topk_list = get_topk(classification_model, mali_update, topk_ratio = 0.9)
    vector_to_parameters(copy.deepcopy(initial_global_model_params), classification_model.parameters())

    training_epoch = 5

    mali_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01, )
    for epoch in range(training_epoch):
        
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            mali_optimizer.zero_grad()
            
            data, target = poison_data_with_normal_trigger(data, target, target_label, poison_frac = 0.2, agent_no = -1)

            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))

            loss.backward()
            apply_grad_mask(classification_model, topk_list)
            
            mali_optimizer.step()

def train_mali_model_with_normal_trigger(classification_model, target_label, agent_train_loader, agent_no = -1):
    classification_model.train()
    training_epoch = 5


    mali_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01, )
    for epoch in range(training_epoch):
        total_loss = 0
        temp_count = 0
        
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            mali_optimizer.zero_grad()
            #0.05 for vgg, 0.2 for resnet
            data, target = poison_data_with_normal_trigger(data, target, target_label, poison_frac = 0.2, agent_no = agent_no)

            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target.view(-1, ))
            
            loss.backward()

            mali_optimizer.step()

            temp_count += 1
            if temp_count % 500 == 0:
              print(loss)

def train_mali_model_with_normal_trigger_htb(classification_model, target_label, agent_train_loader, agent_no = -1, alpha = 0.7):
    initial_global_model_params = parameters_to_vector(classification_model.parameters()).detach()
    target_params_variables = dict()
    for name, param in classification_model.named_parameters():
        target_params_variables[name] = classification_model.state_dict()[name].clone().detach().requires_grad_(False)

    classification_model.train()
    training_epoch = 5

    mali_optimizer = torch.optim.SGD(classification_model.parameters(), lr=0.01, )
    for epoch in range(training_epoch):
        total_loss = 0
        temp_count = 0
        
        for batch_idx, (data, target) in enumerate(agent_train_loader):
            mali_optimizer.zero_grad()
            #0.05 for vgg, 0.2 for resnet
            data, target = poison_data_with_normal_trigger(data, target, target_label, poison_frac = 0.2, agent_no = agent_no)

            output = classification_model(data)

            criterion = nn.CrossEntropyLoss()

            class_loss = criterion(output, target.view(-1, ))
            distance_loss = model_dist_norm_var(classification_model, target_params_variables)

            loss = alpha * class_loss + (1 - alpha) * distance_loss
            loss.backward()

            mali_optimizer.step()
    with torch.no_grad():
            update = parameters_to_vector(classification_model.parameters()).double() - initial_global_model_params
            final_global_model_params = initial_global_model_params + 80 * update
            vector_to_parameters(final_global_model_params, classification_model.parameters())
            