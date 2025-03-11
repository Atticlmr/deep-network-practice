# 配置字典（其他参数）
config_0 = {
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

config_1 = {
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