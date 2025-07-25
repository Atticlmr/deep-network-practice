import torch
import torch.nn as nn

class MLPnet(nn.Module):
    def __init__(self, layer_config, input_size=None, output_activation=None):
        """
        多层感知机(MLP)神经网络
        
        params:
        layer_config:
            
            - [input_size, hidden_size1, hidden_size2, ..., output_size]
            OR
            - [
                {"size": input_size},
                {"size": hidden_size1, "activation": "relu"},
                {"size": hidden_size2, "activation": "tanh", "dropout": 0.2},
                {"size": output_size, "activation": "sigmoid"}
            ]
        
        input_size: int
        output_activation: int
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # simplified [input_size, hidden_size1, ..., output_size]
        if all(isinstance(x, int) for x in layer_config):
            layer_config = [
                {"size": size, "activation": "relu" if i < len(layer_config)-1 else None}
                for i, size in enumerate(layer_config)
            ]
            layer_config[-1]["activation"] = output_activation
        

        if "size" not in layer_config[0]:
            if input_size is None:
                raise ValueError("input size must defined in layer_config OR input_size")
            layer_config[0] = {"size": input_size}
        

        for i in range(1, len(layer_config)):
            prev_size = layer_config[i-1]["size"]
            current_layer = layer_config[i]
            current_size = current_layer["size"]
            
            self.layers.append(nn.Linear(prev_size, current_size))

            activation = current_layer.get("activation", None)
            if activation:
                self.layers.append(self._get_activation(activation))
            

            dropout = current_layer.get("dropout", 0.0)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        
        self.output_activation = self._get_activation(output_activation) if output_activation else None
    
    def _get_activation(self, name):
        """return activarions acording name str"""
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            None: None
        }
        if name.lower() in activations:
            return activations[name.lower()]
        raise ValueError(f"Unsupported activate functions: {name}")
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.output_activation:
            x = self.output_activation(x)
        
        return x
    
# if __name__ == '__main__':
#     import torch
#     from torchinfo import summary
#     simplified_conf = [128,64,32]
#     simplified_model = MLP(simplified_conf)
#     x = torch.randn(4, 128)
#     output = simplified_model(x)

#     print(f"simplified model output shape:{output.shape}")
    
#     mlp_config = [
#         {"size": 128},  
#         {"size": 256, "activation": "relu", "dropout": 0.2},
#         {"size": 128, "activation": "leaky_relu"},
#         {"size": 10, "activation": "sigmoid"} 
#     ]
    
#     model = MLP(mlp_config)
#     output2 = model(x)
#     print(f"complex model output shape{output2.shape}")
#     from utils import logoutput
#     logoutput(model=model, model_conf=mlp_config,depth=6,verbose=1,title="MLP")
    