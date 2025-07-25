import argparse
config_dir= "/config"


from train import train_0,train_1
import json
import torch
from utils import logoutput, params_damp
def parse_args():
    args_cli =argparse.ArgumentParser(description="training")
    args_cli.add_argument('--device',type=str,default='cuda',help='GPU or CPU')
    args_cli.add_argument('--task',type=str,default='',help='experiment name')
    return args_cli.parse_args()


def main():
    args_cli = parse_args()

    if torch.cuda.is_available():
        print('CUDA is available, using GPU for training')
    else:
        print('CUDA not available')

    with open(f'config/CAV.json', 'r') as file:
        configure = json.load(file)
        if args_cli.task == '':
             raise ValueError("task name is None!")
        else:
            base_log_dir = f"./logs/{args_cli.task}"
            LOG_DIR = params_damp(configure=configure, base_log_dir=base_log_dir)

            


    
    
if __name__ == '__main__':
    main()
