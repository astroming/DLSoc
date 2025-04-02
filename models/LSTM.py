import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.Autoformer_EncDec import series_decomp
from layers.multitask import multitask_MLP


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.multitask = configs.mutlitask
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout_ts

        self.dropout_ts = nn.Dropout(configs.dropout_ts)
        self.dropout_static = nn.Dropout(configs.dropout_static)
        self.dropout_lastmlp = nn.Dropout(configs.dropout_lastmlp)

        self.valuelinear = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        ## static feature MLP
        self.static_mlp = nn.ModuleList([nn.Linear(configs.static_in, configs.static_mlp_d, bias=True)] +
                                        [nn.Linear(configs.static_mlp_d, configs.static_mlp_d, bias=True) for _ in
                                         range(configs.static_mlp_layers)])
        ## attentive fusion
        dim_out_ts = configs.d_model * configs.seq_len
        dim_out_static = configs.static_mlp_d
        dim_max = max(dim_out_ts, dim_out_ts)

        self.fusion = nn.ModuleList([nn.Linear(dim_out_ts, dim_max, bias=True)] +
                                    [nn.Linear(dim_out_static, dim_max, bias=True)] +
                                    [nn.Linear(dim_max, dim_max, bias=True)])

        ## last MLP regression
        ## mutli task MLP
        configs.dim_mlpinput = 2 * dim_max
        # ## mutli task MLP
        # configs.dim_mlpinput = configs.d_model * configs.seq_len + configs.static_mlp_d
        self.layers_multitask = multitask_MLP(configs)


    def forward(self, x_enc, x_static, batch_task, mask=None):
        x = self.valuelinear(x_enc)

        # encoding
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(1, x.size(0), self.d_model, device=x.device).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(1, x.size(0), self.d_model, device=x.device).requires_grad_()

        out, (hn, cn) = self.rnn(x,(h0,c0)) # bc,n,d  1,bc,d

        output = out.reshape(out.shape[0], -1)

        # static forward
        x_static = self.static_mlp[0](x_static)
        for linear in self.static_mlp[1:]:
            x_static = F.relu(x_static)
            x_static = self.dropout_static(x_static)
            x_static = linear(x_static)
        # attentive fusion
        x_ts = self.fusion[0](output)
        x_static = self.fusion[1](x_static)
        x_ts_static = x_ts + x_static
        x_ts = self.fusion[2](x_ts_static) * x_ts
        x_static = (1 - self.fusion[2](x_ts_static)) * x_static

        ## concat time series and static for last regression
        x = torch.cat((x_ts, x_static), 1)
        x = self.layers_multitask(x, batch_task)
        return x

