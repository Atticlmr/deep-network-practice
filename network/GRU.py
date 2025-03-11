import torch
import torch.nn as nn

class GRUnet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, fc_layers):
        super(GRUnet, self).__init__()
        self.hidden_layers = hidden_layers
        
        # GRU层
        gru_layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                gru_layers.append(nn.GRU(input_size, hidden_layers[0], batch_first=True))
            else:
                gru_layers.append(nn.GRU(hidden_layers[i-1], hidden_layers[i], batch_first=True))
        self.grus = nn.ModuleList(gru_layers)
        
        # 全连接层
        fc_ = []
        prev_size = hidden_layers[-1]
        for size in fc_layers:
            fc_.append(nn.Linear(prev_size, size))
            fc_.append(nn.ReLU())
            prev_size = size
        fc_.append(nn.Linear(prev_size, output_size))
        self.fcs = nn.Sequential(*fc_)
    
    def forward(self, x):
        # GRU前向传播
        for gru in self.grus:
            h0 = torch.zeros(1, x.size(0), gru.hidden_size).to(x.device)
            x, _ = gru(x, h0)
        
        # 获取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 全连接层前向传播
        x = self.fcs(x)
        return x

if __name__ == '__main__':
    input_size = 10  # 输入特征维度
    output_size = 5  # 输出大小
    hidden_layers = [64, 32]  # 每个GRU层的隐藏单元数列表
    fc_layers = [64, 32]  # 每个全连接层的输出特征数列表
    
    # 创建随机输入数据
    x = torch.randn(32, 10, input_size)  # 形状应为 (batch_size, sequence_length, input_size)
    
    # 实例化模型
    model = GRUnet(input_size, output_size, hidden_layers, fc_layers)
    
    # 前向传播
    try:
        output = model(x)
        print(f'Output shape: {output.shape}')  # 输出形状应为 (batch_size, output_size)
        assert output.shape == (x.size(0), output_size), "Output shape mismatch"
        print("Model is set up correctly.")
    except Exception as e:
        print(f"Error during forward pass: {e}")



