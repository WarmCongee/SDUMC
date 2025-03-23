import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math
import warnings
    
def create_PositionalEncoding(input_dim, max_seq_len=2000): 
    position_encoding = np.array([ 
        [pos / np.power(10000, 2.0 * (j // 2) / input_dim) for j in range(input_dim)] 
        for pos in range(max_seq_len)]) 
    
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
    position_encoding = nn.Parameter(position_encoding, requires_grad=False) 
    
    return position_encoding

def _get_activation_fn(activation: str='relu', module: bool=False):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return nn.ReLU() if module else F.relu
    elif activation == "gelu":
        return nn.GELU() if module else F.gelu
    elif activation == "tanh":
        return nn.Tanh() if module else torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def add_position(x, position=None, mask=None):
    '''add position information to the input x

    x: B, T, C
    position: T, C
    mask: B, T
    '''
    if position is None:
        return x
    else:
        B, T = x.shape[:2]
        position = position[:T].unsqueeze(dim=0).repeat(B, 1, 1)  # -> B, T, C
        position = position*((1 - mask.unsqueeze(-1).type_as(x))) if mask is not None else position
        return x + position

@torch.no_grad()
def arbitrary_segment_mask(start_index: torch.Tensor, end_index: torch.Tensor, len_out: int, reverse: bool = False, return_bool: bool = True):
    '''
    Args:
        start_index: (b t), the first true value
        end_index: (b t), the last true value
        len_out: length of mask 
        reverse: reverse the output mask (Default: False)
        return_bool: if True, return torch.BoolTensor, otherwise torch.FloatTensor
    Returns:
        mask: (b t len_out), the padded values are marked as True if reverse is False
    '''
    b, t = start_index.shape
    start_index = start_index.unsqueeze(dim=-1)  # b t 1
    end_index = end_index.unsqueeze(dim=-1)

    mask = torch.arange(0, len_out, device=start_index.device).unsqueeze(dim=0).unsqueeze(dim=1).expand(b, t, -1)
    mask = ((mask - start_index) < 0) | ((mask - end_index) > 0)

    if reverse:
        mask = ~mask

    if not return_bool:
        mask = mask.float()

    return mask

@torch.no_grad()
def arbitrary_point_mask(index: torch.Tensor, len_out: int, reverse: bool = False, return_bool: bool = True):
    '''
    Args:
        index: (b t), index of the masked point
        len_out: length of mask 
        reverse: reverse the output mask (Default: False)
        return_bool: if True, return torch.BoolTensor, otherwise torch.FloatTensor
    Returns:
        mask: (b t len_out), the specified points are marked as True if reverse is False
    '''
    b, t = index.shape
    index = index.unsqueeze(dim=-1)  # b t 1

    mask = torch.arange(0, len_out, device=index.device).unsqueeze(dim=0).unsqueeze(dim=1).expand(b, t, -1)
    mask = (mask - index) == 0

    if reverse:
        mask = ~mask

    if not return_bool:
        mask = mask.float()

    return mask

def inverse_sigmoid(y):
    assert 0 < y < 1, f'y out of range (0 < y < 1), current y = {y}'
    return math.log(y/(1-y))

class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, dropout=0., activation='relu'):
        super().__init__()
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)

        return x

class Deformable_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.length = length
        self.init_duration_min = duration_init_min
        self.init_duration_max = duration_init_max

        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5

        self.offset_duration = nn.Linear(embed_dim, num_heads * 2, bias=True)
        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias)
    
    def _reset_specially(self):
        nn.init.constant_(self.offset_duration.weight.data, 0.)

        offset_init = torch.zeros(self.num_heads).type(torch.float)
        duration_init = torch.arange(self.init_duration_min, self.init_duration_max+1, \
            step=(self.init_duration_max-self.init_duration_min)/(self.num_heads-1)) / self.length

        duration_init = torch.tensor(list(map(inverse_sigmoid, duration_init)))

        bias_init = torch.cat((offset_init, duration_init), dim=-1)
        with torch.no_grad():
            self.offset_duration.bias = nn.Parameter(bias_init.view(-1))

    def query_index(self, bsz, dtype, device):
        return torch.arange(0, self.length, dtype=dtype, device=device)[None, :, None].expand(bsz, -1, -1)

    def forward(self, x: torch.Tensor):
        # b t c
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        print(x.shape)
        print(Q.shape)

        # b t nh
        offset_duration = self.offset_duration(Q) #* self.length
        offset, duration = offset_duration.chunk(2, dim=-1)
        print(offset.shape)
        offset = offset.tanh() * self.length
        duration = duration.sigmoid() * self.length

        query_index = self.query_index(x.shape[0], x.dtype, x.device)
        # print(query_index.shape)
        print(offset.shape)
        anchor = query_index + offset

        # (b nh) t
        anchor = einops.rearrange(anchor, 'b t nh -> (b nh) t', nh=self.num_heads)
        duration = einops.rearrange(duration, 'b t nh -> (b nh) t', nh=self.num_heads)#.abs_()

        # (b nh) t
        start = anchor - duration
        end = anchor + duration
        bound_l_idx_l = torch.floor(start)
        bound_r_idx_r = torch.ceil(end)
        anchor_idx_l = torch.floor(anchor)
        anchor_idx_r = anchor_idx_l + 1

        # (b nh) t
        neg_bound_l_dist_l = bound_l_idx_l - start
        neg_bound_r_dist_r = end - bound_r_idx_r
        anchor_dist_l = anchor - anchor_idx_l
        anchor_dist_r = 1 - anchor_dist_l
        
        # (b nh) t t   requires_grad is False
        bound_l_mask_l = arbitrary_point_mask(index=bound_l_idx_l, len_out=self.length, return_bool=False)
        bound_r_mask_r = arbitrary_point_mask(index=bound_r_idx_r, len_out=self.length, return_bool=False)
        anchor_l_mask = arbitrary_point_mask(index=anchor_idx_l, len_out=self.length, return_bool=False)
        anchor_r_mask = arbitrary_point_mask(index=anchor_idx_r, len_out=self.length, return_bool=False)
        
        # (b nh) t t
        neg_bound_weight = neg_bound_l_dist_l[:, :, None] * bound_l_mask_l + neg_bound_r_dist_r[:, :, None] * bound_r_mask_r
        anchor_weight = anchor_dist_l[:, :, None] * anchor_r_mask + anchor_dist_r[:, :, None] * anchor_l_mask
        point_weight = 1 + neg_bound_weight + anchor_weight

        # (b nh) t t
        attn_mask = arbitrary_segment_mask(start_index=bound_l_idx_l, end_index=bound_r_idx_r, len_out=self.length)

        # (b nh) t hd
        Q = einops.rearrange(Q, 'b t (nh hd) -> (b nh) t hd', nh=self.num_heads)
        K = einops.rearrange(K, 'b t (nh hd) -> (b nh) t hd', nh=self.num_heads)
        V = einops.rearrange(V, 'b t (nh hd) -> (b nh) t hd', nh=self.num_heads)

        # (b nh) t t
        attn_weights = torch.einsum('...ic, ...rc -> ...ir', Q, K) * self.scaling
        attn_weights = attn_weights * point_weight
        attn_weights = attn_weights.masked_fill(attn_mask, -1e8)  # float('-inf')
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('...ir, ...rc -> ...ic', attn_weights, V)
        output = einops.rearrange(output, '(b nh) t hd -> b t (nh hd)', nh=self.num_heads)

        # b t c
        output = self.project_out(output)

        return output

class DST_Encoder(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout=0., bias=True, activation='relu'):
        super().__init__()
        self.dropout = dropout

        self.attention = Deformable_Attention(embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout, bias)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_embed_dim, dropout, activation)

    def perform_attn(self, x):
        residual = x
        x = self.attention(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_norm(x)
        return x

    def perform_ffn(self, x):
        return self.ffn(x)

    def forward(self, x):
        x = self.perform_attn(x)
        x = self.perform_ffn(x)
        return x


class DST_ATT(nn.Module):
    def __init__(self, args,
                input_dim       = 1024,
                length          = 326,
                ffn_embed_dim   = 512, 
                num_layers      = 4, 
                num_heads       = 8, 
                num_classes     = 1, 
                dropout         = 0.1,
                bias            = True,
                activation      = 'relu'):
        super().__init__()
        
        self.position = create_PositionalEncoding(input_dim)
        self.input_norm = nn.LayerNorm(input_dim)

        hop = 20
        duration = [50, 1000]
        duration_init_min = duration[0] / hop / 2  # half of duration: / 2
        duration_init_max = duration[1] / hop / 2
        
        self.layers = nn.ModuleList(
            [DST_Encoder(input_dim, ffn_embed_dim, num_heads, length, duration_init_min, duration_init_max, dropout, bias, activation) 
            for _ in range(num_layers)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            _get_activation_fn(activation, module=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//2, input_dim//4),
            _get_activation_fn(activation, module=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim//4, num_classes),
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for m in self.modules():
            if hasattr(m, '_reset_specially'):
                m._reset_specially()

    def forward(self, x):
        x = self.input_norm(x[0])
        x = add_position(x, self.position)

        for layer in self.layers:
            x = layer(x)
        
        pred = self.classifier(x)
        pred = self.avgpool(pred.transpose(-1, -2)).squeeze(dim=-1)
        _ = []
        return x, pred, _