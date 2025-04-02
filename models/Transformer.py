import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.multitask import multitask_MLP
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        ## dropout
        self.dropout_ts = nn.Dropout(configs.dropout_ts)
        self.dropout_static = nn.Dropout(configs.dropout_static)
        self.dropout_lastmlp = nn.Dropout(configs.dropout_lastmlp)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout_ts)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout_ts,
                                      output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_enconv,
                    dropout=configs.dropout_ts,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        ## static feature MLP
        self.static_mlp = nn.ModuleList([nn.Linear(configs.static_in, configs.static_mlp_d, bias=True)]+
            [nn.Linear(configs.static_mlp_d, configs.static_mlp_d, bias=True) for _ in range(configs.static_mlp_layers)])
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
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        output, attns = self.encoder(enc_out, attn_mask=None)

        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
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
