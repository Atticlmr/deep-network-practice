import argparse
config_dir= "/config"
from network.CNN import CNNnet
from train import train_0,train_1,evaluate
import json
import torch

def parse_args():
    args_cli =argparse.ArgumentParser(description="CNN_training")
    args_cli.add_argument('--device',type=str,default='cuda',help='GPU or CPU')
    args_cli.add_argument('--tasks',type=str,default='treasure',help='treasure:宝石分类 cat_dog:猫狗分类')

    return args_cli.parse_args()


def main():
    args_cli = parse_args()

    if torch.cuda.is_available():
        print('CUDA is available')
    else:
        print('CUDA not available')
    with open(f'config/{args_cli.tasks}.json', 'r') as file:
        configure = json.load(file)
    
    if args_cli.tasks == 'treasure':
        model = CNNnet(num_classes=configure['num_classes'],conv_layers_config = configure['conv_layers_config']
                       ).to(args_cli.device)
        train_0(args_cli, model, configure)
    elif args_cli.tasks == "cat_dog":
        model = CNNnet(num_classes=configure['num_classes'],conv_layers_config = configure['conv_layers_config']
                ).to(args_cli.device)
        train_1(args_cli, model, configure)

    
if __name__ == '__main__':
    main()
