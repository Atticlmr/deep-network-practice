
import torch.nn as nn
import torch.nn.functional as F


class CNNnet(nn.Module):
    def __init__(self,
                   in_channels=3,
            num_classes=10,
            conv_layers_config=[
                     {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                     {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
                 ],
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_size=128,
            dropout_rate=0.0):
        super(CNNnet,self).__init__()

        # 动态创建卷积层
        self.conv_layers = nn.ModuleList()
        prev_channels = in_channels
        for layer_config in conv_layers_config:
            # 卷积层
            conv = nn.Conv2d(
                prev_channels,
                layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding']
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            # Dropout
            if dropout_rate > 0:
                self.conv_layers.append(nn.Dropout2d(p=dropout_rate))
            
            prev_channels = layer_config['out_channels']


        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
        fc_input_size = prev_channels
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)
        

        x = self.global_pool(x)
        

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
