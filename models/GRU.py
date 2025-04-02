import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.multitask import multitask_MLP
# from layers.Autoformer_EncDec import series_decomp


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

        self.pred_len = configs.seq_len

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        ## static feature MLP
        self.static_mlp = nn.ModuleList([nn.Linear(configs.static_in, configs.static_mlp_d, bias=True)] +
                                        [nn.Linear(configs.static_mlp_d, configs.static_mlp_d, bias=True) for _ in
                                         range(configs.static_mlp_layers)])
        ## attentive fusion
        dim_out_ts = configs.enc_in * configs.seq_len
        dim_out_static = configs.static_mlp_d
        dim_max = max(dim_out_ts, dim_out_ts)

        self.fusion = nn.ModuleList([nn.Linear(dim_out_ts, dim_max, bias=True)] +
                                    [nn.Linear(dim_out_static, dim_max, bias=True)] +
                                    [nn.Linear(dim_max, dim_max, bias=True)])

        ## last MLP regression
        ## mutli task MLP
        configs.dim_mlpinput = 2 * dim_max
        # ## mutli task MLP
        # configs.dim_mlpinput = configs.enc_in * configs.seq_len + configs.static_mlp_d
        self.layers_multitask = multitask_MLP(configs)




    def forward(self, x_enc, x_static, batch_task, mask=None):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x_enc.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x_enc[:, -1:, :].detach()
        x = (x_enc - seq_last).permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        enc_out = y.permute(0, 2, 1) + seq_last
        output = enc_out.reshape(enc_out.shape[0], -1)

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

