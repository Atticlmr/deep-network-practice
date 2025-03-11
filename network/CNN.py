
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

if __name__ == '__main__':
    import torch
    # 参数设置
    in_channels = 3  # 输入通道数（例如RGB图像为3）
    num_classes = 10  # 分类数量
    conv_layers_config = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    hidden_size = 128  # 全连接层隐藏单元数
    dropout_rate = 0.0  # Dropout率
    
    # 创建随机输入数据
    batch_size = 4  # 批处理大小
    image_height = 32  # 图像高度
    image_width = 32  # 图像宽度
    x = torch.randn(batch_size, in_channels, image_height, image_width)  # 形状应为 (batch_size, in_channels, height, width)
    
    # 实例化模型
    model = CNNnet(in_channels=in_channels, num_classes=num_classes, conv_layers_config=conv_layers_config, hidden_size=hidden_size, dropout_rate=dropout_rate)
    
    # 前向传播
    try:
        output = model(x)
        print(f'Output shape: {output.shape}')  # 输出形状应为 (batch_size, num_classes)
        assert output.shape == (x.size(0), num_classes), "Output shape mismatch"
        print("Model is set up correctly.")
    except Exception as e:
        print(f"Error during forward pass: {e}")


