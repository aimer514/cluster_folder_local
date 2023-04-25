from option import *
from data_loader import *
global num_of_malicious
global device
global using_wandb
from Aggregation import *

def trigger_generation_train(temp_model, noise_model, train_loader_list, test_loader, args):
    init_sparsefed(temp_model)
    init_foolsgold(temp_model)
    total_epoch = args.total_epoch  
    target_label = args.target_label
    possible = args.possibility
    aggregation_dict = {}
    use_heatmap = False
    norm_for_one_sample = args.trigger_norm
    batch_norm_list = get_batch_norm_list(temp_model)
    unet_batch_norm_list = get_batch_norm_list(noise_model)

    agent_batch_norm_list = initialize_batch_norm_list(temp_model, batch_norm_list)
    unet_agent_batch_norm_list = initialize_batch_norm_list(noise_model, unet_batch_norm_list)

    if using_wandb:
        wandb.init(project= args.wandb_project_name, name = args.wandb_run_name, entity="harrychen23235")

    for epoch_num in range(total_epoch):
        rnd_batch_norm_dict = {}
        print('current epoch is {}'.format(epoch_num))
        start_parameter = parameters_to_vector(temp_model.parameters()).detach()
        save_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)
        save_batch_norm(noise_model, 0, unet_batch_norm_list, unet_agent_batch_norm_list)

        aggregation_dict = {}
        rnd_num = random.random()
        if args.save_checkpoint_path is not None:
            if epoch_num % 5 == 0:
                torch.save(temp_model.state_dict(), args.save_checkpoint_path + '/rnd_{}_model.pt'.format(epoch_num))
                torch.save(agent_batch_norm_list[0], args.save_checkpoint_path + 'rnd_{}_bn.pt'.format(epoch_num))
                torch.save(noise_model.state_dict(), args.save_checkpoint_path + 'rnd_{}_unet.pt'.format(epoch_num))  
                torch.save(unet_agent_batch_norm_list[0], args.save_checkpoint_path + 'rnd_{}_unet_bn.pt'.format(epoch_num))

        if using_wandb:
            if rnd_num < possible:
                wandb.log({'attack_inside':1})
            else:
                wandb.log({'attack_inside':0})

        if epoch_num >= 0 and rnd_num < possible:
            noise_model = train_noise_model(temp_model, target_label, train_loader_list[0], norm_for_one_sample = norm_for_one_sample, use_heatmap = use_heatmap, input_noise_model = noise_model)

        for agent in range(num_of_agent):
            #print('current agent is')
            #print(agent)
            load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)
            if agent < num_of_malicious and epoch_num >= 0 and rnd_num < possible:
                train_mali_model_with_noise(temp_model, noise_model, target_label, train_loader_list[agent], norm_for_one_sample, use_heatmap = use_heatmap)
            else:
                train_benign_model(temp_model,train_loader_list[agent])

            with torch.no_grad():
                local_model_update_dict = dict()
                for name, data in temp_model.state_dict().items():
                    if name in batch_norm_list:
                        local_model_update_dict[name] = torch.zeros_like(data)
                        local_model_update_dict[name] = (data - agent_batch_norm_list[0][name])
                rnd_batch_norm_dict[agent] = local_model_update_dict

            with torch.no_grad():
                temp_update = parameters_to_vector(temp_model.parameters()).double() - start_parameter
            
            aggregation_dict[agent] = temp_update
            vector_to_parameters(copy.deepcopy(start_parameter), temp_model.parameters())

        if epoch_num >= 0 and rnd_num < possible and using_wandb:
            wandb.log({'mali_norm':torch.norm(aggregation_dict[0]).item()})

        if args.using_clip:
            clip = get_average_norm(aggregation_dict)
        else:
            clip = 0

        if using_wandb:
            wandb.log({'average_clip':clip})

        load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)

        benign_list = aggregation_time(temp_model, aggregation_dict, clip = clip, agg_way = args.aggregation)
        aggregate_batch_norm(temp_model, rnd_batch_norm_dict)

        benign_accuracy = test_model(temp_model, test_loader)
        malicious_accuracy = test_mali_noise(temp_model, noise_model, test_loader, target_label = target_label, norm_bound = norm_for_one_sample, use_heatmap = use_heatmap)
        if using_wandb:
            wandb.log({"mali_acc": malicious_accuracy, "benign_accuracy": benign_accuracy})

    if using_wandb:
        wandb.finish()


def normal_train(temp_model, train_loader_list, test_loader, args):
    init_sparsefed(temp_model)
    init_foolsgold(temp_model)
    total_epoch = args.total_epoch  
    target_label = args.target_label
    possible = args.possibility
    aggregation_dict = {}
    use_heatmap = False

    batch_norm_list = get_batch_norm_list(temp_model)
    agent_batch_norm_list = initialize_batch_norm_list(temp_model, batch_norm_list)


    if using_wandb:
        wandb.init(project= args.wandb_project_name, name = args.wandb_run_name, entity="harrychen23235")

    for epoch_num in range(total_epoch):
        rnd_batch_norm_dict = {}
        print('current epoch is {}'.format(epoch_num))
        start_parameter = parameters_to_vector(temp_model.parameters()).detach()
        save_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)

        aggregation_dict = {}
        rnd_num = random.random()
        if args.save_checkpoint_path is not None:
            if epoch_num % 5 == 0:
                torch.save(temp_model.state_dict(), args.save_checkpoint_path + '/rnd_{}_model.pt'.format(epoch_num))
                torch.save(agent_batch_norm_list[0], args.save_checkpoint_path + 'rnd_{}_bn.pt'.format(epoch_num))

        if using_wandb:
            if rnd_num < possible:
                wandb.log({'attack_inside':1})
            else:
                wandb.log({'attack_inside':0})

        for agent in range(num_of_agent):
            #print('current agent is')
            #print(agent)
            load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)
            if agent < num_of_malicious and epoch_num >= 0 and rnd_num < possible:
                if attack_mode == 'DBA':
                    train_mali_model_with_normal_trigger(temp_model, target_label, train_loader_list[agent], agent_no = random.randint(0,3))
                    
                elif attack_mode == 'durable':
                    train_mali_model_with_normal_trigger_topk_mode(temp_model, target_label, train_loader_list[agent])
                else:
                    train_mali_model_with_normal_trigger(temp_model, target_label, train_loader_list[agent])
            else:
                train_benign_model(temp_model,train_loader_list[agent])

            with torch.no_grad():
                local_model_update_dict = dict()
                for name, data in temp_model.state_dict().items():
                    if name in batch_norm_list:
                        local_model_update_dict[name] = torch.zeros_like(data)
                        local_model_update_dict[name] = (data - agent_batch_norm_list[0][name])
                rnd_batch_norm_dict[agent] = local_model_update_dict

            with torch.no_grad():
                temp_update = parameters_to_vector(temp_model.parameters()).double() - start_parameter
            
            aggregation_dict[agent] = temp_update
            vector_to_parameters(copy.deepcopy(start_parameter), temp_model.parameters())

        if epoch_num >= 0 and rnd_num < possible and using_wandb:
            wandb.log({'mali_norm':torch.norm(aggregation_dict[0]).item()})

        if args.using_clip:
            clip = get_average_norm(aggregation_dict)
        else:
            clip = 0

        if using_wandb:
            wandb.log({'average_clip':clip})

        load_batch_norm(temp_model, 0, batch_norm_list, agent_batch_norm_list)

        benign_list = aggregation_time(temp_model, aggregation_dict, clip = clip, agg_way = args.aggregation)
        aggregate_batch_norm(temp_model, rnd_batch_norm_dict)

        benign_accuracy = test_model(temp_model, test_loader)
        malicious_accuracy = test_mali_normal_trigger(temp_model, test_loader, target_label)
        if using_wandb:
            wandb.log({"mali_acc": malicious_accuracy, "benign_accuracy": benign_accuracy})

    if using_wandb:
        wandb.finish()

def config_global_variable(args):
    import Aggregation
    import AutoEncoder
    import Unet
    Aggregation.device = args.device
    Aggregation.num_of_agent = args.num_of_agent
    Aggregation.using_wandb = args.if_wandb
    Aggregation.num_of_malicious = args.num_of_malicious
    Aggregation.lr = args.server_lr
    AutoEncoder.device = args.device
    Unet.device = args.device

if __name__ == '__main__':
    args = args_parser()
    # args.if_wandb = True
    # args.wandb_project_name = 'test_local'
    # args.wandb_run_name = 'test_local'

    device = args.device
    num_of_malicious = args.num_of_malicious
    dataset = args.dataset
    num_of_agent = args.num_of_agent
    iid = args.iid
    using_wandb = args.if_wandb
    attack_mode = args.attack_mode


    config_global_variable(args)
    print("args is")
    print(args)
    if using_wandb:
        wandb.login(key = '40d461d04db022d2a1945f31ee4a36c90708e9a4')

    if dataset == "cifar10":
        from cifar10_train import *
    elif dataset == "tiny":
        from tiny_train import *
    
    #dataset loading
    train_dataset, test_dataset = load_dataset(dataset, args.dataset_path)

    test_loader = test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)

    if dataset == "tiny":
        n_classes = 200
    else:
        n_classes = 10

    train_loader_list = split_train_data(train_dataset, num_of_agent = num_of_agent, non_iid = not iid, n_classes= n_classes)

    if dataset == "cifar10":
        temp_model = ResNet18(name = 'local').to(device)
    elif dataset == "tiny":
        temp_model = resnet18(name = 'local').to(device = device)

    if attack_mode == 'trigger_generation':
        if dataset == "cifar10":
            noise_model = UNet(3).to(device = device)
        elif dataset == "tiny":
            noise_model = Autoencoder().to(device = device)
    
    if attack_mode == 'trigger_generation':
        trigger_generation_train(temp_model, noise_model, train_loader_list, test_loader, args)
    else:
        normal_train(temp_model, train_loader_list, test_loader, args)