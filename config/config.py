# 配置字典（其他参数）
treasure = {
    'task': 'treasure',
    'data_dir': '/home/li/Desktop/code/CV/dataset/archive_train',
    'test_dir': '/home/li/Desktop/code/CV/dataset/archive_test',

    'num_classes': 25,
    'batch_size': 32,
    'lr': 0.001,
    'num_epochs': 100,
    'log_dir':"logs/treasure",
    'save_dir': '/home/li/Desktop/code/CV/logs/model/treasure',
    'image_size': 224,
    'early_stop_patience': 7,
    'conv_layers_config':[
                     {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                     {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
                 ]
}

cifar = {
    'task': 'cifat-10',
    'data_dir': '/home/li/Desktop/code/CV/dataset/cifar-10-python',
    'num_classes': 10,
    'batch_size': 32,
    'lr': 0.001,
    'num_epochs': 100,
    'log_dir':"logs/cat_dog",
    'save_dir': '/home/li/Desktop/code/CV/logs/model/cat_dog',
    'image_size': 224,
    'early_stop_patience': 7,
    'conv_layers_config' : [
    {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
]
}
import json
def write_conf(conf_name,conf):
# 将配置写入JSON文件
    with open(f'config/{conf_name}.json', 'w') as f:
        json.dump(conf, f, ensure_ascii=False, indent=4)
import argparse
if __name__ == "__main__":
    args_cli =argparse.ArgumentParser(description="config write")
    args_cli.add_argument('--task',type=str,default='task',help='task name')

    write_conf(args_cli.parse_args().task)