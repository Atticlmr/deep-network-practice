import argparse
import config.config as config
from network.CNN import CNNnet
from train import train_0,train_1,evaluate
import torch

def parse_args():
    args_cli =argparse.ArgumentParser(description="CNN_training")
    args_cli.add_argument('--device',type=str,default='cuda',help='GPU or CPU')
    args_cli.add_argument('--tasks',type=int,default=0,help='0:宝石分类 1:猫狗分类')

    return args_cli.parse_args()


def main():
    args_cli = parse_args()

    if args_cli.device == 'cuda':
        print(torch.cuda.is_available())

        print('CUDA available')
    else:
        print('CUDA not available')

    if args_cli.tasks == 0:
        configure = config.config_0
    elif args_cli.tasks == 1:
        configure = config.config_1
    
    if args_cli.tasks == 0:
        model = CNNnet(num_classes=configure['num_classes'],conv_layers_config = configure['conv_layers_config']
                       ).to(args_cli.device)
        train_0(args_cli, model, configure)
    elif args_cli.tasks == 1:
        model = CNNnet(num_classes=configure['num_classes'],conv_layers_config = configure['conv_layers_config']
                ).to(args_cli.device)
        train_1(args_cli, model, configure)

    
if __name__ == '__main__':
    main()
