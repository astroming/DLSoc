import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import weight_norm
# import math


class multitask_MLP(nn.Module):
    def __init__(self, configs):
        super(multitask_MLP, self).__init__()
        self.multitask = configs.mutlitask
        ## dropout
        self.dropout_lastmlp = nn.Dropout(configs.dropout_lastmlp)
        ## last MLP regression
        self.mlp = nn.ModuleList(
            [nn.Linear(configs.dim_mlpinput, configs.mlp_layers_d, bias=True)] +
            [nn.Linear(configs.mlp_layers_d, configs.mlp_layers_d, bias=True) for _ in range(configs.mlp_layers)] +
            [nn.Linear(configs.mlp_layers_d, 64, bias=True)] +
            [nn.Linear(64, 1, bias=True)])
        ## mutli task MLP
        self.sharemlp1 = nn.ModuleList([nn.Linear(configs.dim_mlpinput, \
                                                  configs.mlp_layers_d, bias=True)] + \
                                       [nn.Linear(configs.mlp_layers_d, configs.mlp_layers_d, bias=True) \
                                        for _ in range(configs.mlp_layers_shared1)])

        self.sharemlp2 = nn.ModuleList([nn.Linear(configs.mlp_layers_d, configs.mlp_layers_d, bias=True) \
                                        for _ in range(configs.mlp_layers_shared2)])
        task1 = {}
        for task in configs.tasks:
            task1[str(task)] = nn.ModuleList([nn.Linear(configs.mlp_layers_d, configs.mlp_layers_d, bias=True) \
                                              for _ in range(configs.mlp_layers_task1)])
        self.task1 = nn.ModuleDict(task1)

        task2 = {}
        for task in configs.tasks:
            task2[str(task)] = nn.ModuleList([nn.Linear(configs.mlp_layers_d, configs.mlp_layers_d, bias=True) \
                                              for _ in range(configs.mlp_layers_task2)] + \
                                             [nn.Linear(configs.mlp_layers_d, 64, bias=True)] + \
                                             [nn.Linear(64, 1, bias=True)])
        self.task2 = nn.ModuleDict(task2)

    def forward(self, x, batch_task):
        if self.multitask:
            # shared 1 layers
            x = self.sharemlp1[0](x)
            for linear in self.sharemlp1[1:]:  # changed
                x = F.relu(x)
                x = self.dropout_lastmlp(x)
                x = linear(x)

            # task 1 layers
            temp = []
            for idx, task in enumerate(batch_task):
                row = x[idx]  # .unsqueeze(1).T
                module_key = str(task[0])
                for linear in self.task1[module_key]:
                    row = F.relu(row)
                    row = self.dropout_lastmlp(row)
                    row = linear(row)
                temp.append(row)
            x = torch.stack(temp, dim=0)

            # shared 2 layers
            x = self.sharemlp2[0](x)
            for linear in self.sharemlp2[1:]:  # changed
                x = F.relu(x)
                x = self.dropout_lastmlp(x)
                x = linear(x)

            # task 2 layers
            temp = []
            for idx, task in enumerate(batch_task):
                row = x[idx]
                module_key = str(task[0])
                for linear in self.task2[module_key][:-1]:
                    row = F.relu(row)
                    row = self.dropout_lastmlp(row)
                    row = linear(row)
                row = self.task2[module_key][-1](row)
                temp.append(row)
            x = torch.stack(temp, dim=0)
        else:
            x = self.mlp[0](x)
            for linear in self.mlp[1:-2]:  # changed
                x = F.relu(x)
                x = self.dropout_lastmlp(x)
                x = linear(x)
            x = self.mlp[-2](x)
            x = self.mlp[-1](x)
        x = torch.sigmoid(x)

        return x

