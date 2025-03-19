import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out


class CNNnetWithResiduals(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=10,
                 conv_layers_config=[
                     {'out_channels': 32, 'stride': 1},
                     {'out_channels': 64, 'stride': 2}
                 ],
                 hidden_size=128,
                 dropout_rate=0.0):
        super(CNNnetWithResiduals, self).__init__()

        # 初始卷积层
        self.initial_conv = nn.Conv2d(in_channels, conv_layers_config[0]['out_channels'], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(conv_layers_config[0]['out_channels'])
        
        # 动态创建残差块
        self.residual_blocks = nn.ModuleList()
        current_in_channels = conv_layers_config[0]['out_channels']  # 初始化为第一个块的输入通道数
        for layer_config in conv_layers_config:
            out_channels = layer_config['out_channels']
            stride = layer_config['stride']
            downsample = None
            if (stride != 1) or (current_in_channels != out_channels):
                downsample = nn.Sequential(
                    nn.Conv2d(current_in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels))
            self.residual_blocks.append(ResidualBlock(current_in_channels, out_channels, stride, downsample))
            current_in_channels = out_channels  # 更新输入通道数为当前块的输出通道数

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        fc_input_size = conv_layers_config[-1]['out_channels']
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn(x)
        x = F.relu(x)

        for block in self.residual_blocks:
            x = block(x)

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
        {'out_channels': 32, 'stride': 1},
        {'out_channels': 64, 'stride': 2}
    ]
    hidden_size = 128  # 全连接层隐藏单元数
    dropout_rate = 0.0  # Dropout率
    
    # 创建随机输入数据
    batch_size = 4  # 批处理大小
    image_height = 32  # 图像高度
    image_width = 32  # 图像宽度
    x = torch.randn(batch_size, in_channels, image_height, image_width)  # 形状应为 (batch_size, in_channels, height, width)
    
    # 实例化模型
    model = CNNnetWithResiduals(in_channels=in_channels, num_classes=num_classes, conv_layers_config=conv_layers_config, hidden_size=hidden_size, dropout_rate=dropout_rate)
    
    # 前向传播
    try:
        output = model(x)
        print(f'Output shape: {output.shape}')  # 输出形状应为 (batch_size, num_classes)
        assert output.shape == (x.size(0), num_classes), "Output shape mismatch"
        print("Model is set up correctly.")
    except Exception as e:
        print(f"Error during forward pass: {e}")