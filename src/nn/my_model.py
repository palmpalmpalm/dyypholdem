import torch
from torch import nn 

ip_size = 1009
op_size = 1008

# Baseline model from the paper

class BaselineNN(nn.Module):
    def __init__(self, hidden_size=500,input_size=ip_size,output_size=op_size):
        super(BaselineNN, self).__init__()
        
        model = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.PReLU(),
          nn.Linear(hidden_size, hidden_size),
          nn.PReLU(),
          nn.Linear(hidden_size, hidden_size),
          nn.PReLU(),
          nn.Linear(hidden_size, hidden_size),
          nn.PReLU(),
          nn.Linear(hidden_size, output_size),
        )

        self.model = model
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        # Normal ff
        feedforward = self.model(x)

        # Zerosum part
        ranges = torch.narrow(x,2, 0, self.output_size)
        batch1 = feedforward
        batch2 = torch.moveaxis(ranges,1,2)
        estimated_value = torch.bmm(batch1, batch2).squeeze(2)        
        estimated_value = estimated_value.repeat(1, self.output_size).unsqueeze(1)
        estimated_value = torch.mul(estimated_value, -0.5)
        final_mlp = torch.add(feedforward, estimated_value)

        return final_mlp
    
