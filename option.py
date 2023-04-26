import argparse
import torch

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_agent', type=int, default=10,
                        help="number of agents:K")
    
    parser.add_argument('--num_of_malicious', type=int, default=1,
                    help="number of corrupt agents")
    
    parser.add_argument('--attack_mode', type=str, default='base',
                    help="trigger_generation, base, DBA, durable")
    
    parser.add_argument('--total_epoch', type=int, default=200,
                help="number of total epochs")
    
    parser.add_argument('--possibility', type=float, default=0.25,
            help="possibility of selecting malicious agents")
    
    parser.add_argument('--target_label', type=int, default=7,
            help="target label index")
    
    parser.add_argument('--aggregation', type=str, default="avg",
        help="aggregation method")
    
    parser.add_argument('--using_clip', type = boolean_string, default=False,
                        help="average clip or not")
    
    parser.add_argument('--iid', type = boolean_string, default=True,
                    help="iid or not")
    
    parser.add_argument('--save_checkpoint_path', type = str, default = None,
                help="path of saving checkpoint")
    
    parser.add_argument('--if_wandb', type = boolean_string, default=False,
                help="wandb or not")
    
    
    parser.add_argument('--wandb_project_name', type = str, default="",
            help="wandb project name")
    
    parser.add_argument('--wandb_run_name', type = str, default="",
        help="wandb run name")
    
    
    parser.add_argument('--dataset_path', type = str, default="./data",
                    help="path of dataset")

    parser.add_argument('--pretrained_checkpoint_path', type = str, default=None,
                help="path of pretrained checkpoint")
    
    parser.add_argument('--pretrained_checkpoint_path_batch_norm', type = str, default=None,
            help="path of pretrained checkpoint of batch norm")

    parser.add_argument('--trigger_norm', type=int, default=3,
        help="norm of trigger")
    
    parser.add_argument('--device', type=str, default="cuda:0",
    help="device of training")
    
    parser.add_argument('--dataset', type=str, default="cifar10",
    help="cifar10, tiny")

    parser.add_argument('--server_lr', type=float, default=1,
    help="lr of server")

    args = parser.parse_args()
    return args
