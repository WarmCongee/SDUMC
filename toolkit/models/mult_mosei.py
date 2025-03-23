"""
From: https://github.com/yaohungt/Multimodal-Transformer
Paper: Multimodal Transformer for Unaligned Multimodal Language Sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.transformers_encoder.transformer import TransformerEncoder


class FRA2UTT_new(nn.Module):
    def __init__(self, input_dim=1024, atsize=1024, softmax_scale=0.3):
        super(FRA2UTT_new, self).__init__()
        self.atsize = atsize
        self.softmax_scale = softmax_scale
        self.attention_context_vector = nn.Parameter(torch.empty(1, input_dim)) #(batch_size, feature_dim)
        nn.init.xavier_normal_(self.attention_context_vector)
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.dropout_output = nn.Dropout(0.5)
    
    def forward(self, input_tensor):
        input_tensor = self.dropout_output(input_tensor)
        batch_size = input_tensor.shape[0]
        attention_context_vector = self.attention_context_vector.repeat(batch_size,1).unsqueeze(2)
        input_proj = torch.tanh(self.input_proj(input_tensor))
        vector_attention = torch.bmm(input_proj, attention_context_vector)
        #softmax
        vector_attention = F.softmax(self.softmax_scale*vector_attention,dim=1)
        output_vector = torch.mul(input_tensor, vector_attention)
        output_vector.squeeze()
        output_tensor = torch.sum(output_vector, dim=1, keepdim=False)
        output_tensor = self.dropout_output(output_tensor)
        return output_tensor


class MULTMOSEI(nn.Module):
    def __init__(self, args):
        super(MULTMOSEI, self).__init__()

        # params: analyze args
        audio_dim   = args.input_dims[0]
        text_dim    = args.input_dims[1]
        video_dim   = args.input_dims[2]
        output_dim1 = 8 # args.output_dim1
        output_dim2 = 1 # args.output_dim1

        # params: analyze args
        self.attn_mask = True
        self.layers = 4 # args.layers # 4 
        self.dropout = args.dropout
        self.num_heads = 2 # args.num_heads # 8
        self.hidden_dim = 128 # args.hidden_dim # 128
        self.conv1d_kernel_size = 5 # args.conv1d_kernel_size # 5
        # self.grad_clip = args.grad_clip
        
        # params: intermedia
        combined_dim = 5*self.hidden_dim # 2 * (self.hidden_dim + self.hidden_dim + self.hidden_dim)
        output_dim = self.hidden_dim // 2
        self.dropout_output = nn.Dropout(0.5)


        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(video_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_l_mem = FRA2UTT_new(2*self.hidden_dim)
        self.trans_l_self = FRA2UTT_new(self.hidden_dim)
        self.trans_a_self = FRA2UTT_new(self.hidden_dim)
        self.trans_v_self = FRA2UTT_new(self.hidden_dim)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # cls layers
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        else:
            raise ValueError("Unknown network type")
        
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, batch):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            video_feat: tensor of shape (batch, seqlen2, video_in)
            text_feat:  tensor of shape (batch, seqlen3, text_in)
        '''
        x_a = batch[0].transpose(1, 2)  # 'bt dim l'
        x_l = batch[1].transpose(1, 2)
        x_v = batch[2].transpose(1, 2)

        x_l = self.dropout_output(x_l)
        x_a = self.dropout_output(x_a)
        x_v = self.dropout_output(x_v)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1) # 'l bt dim'
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        proj_x_l = self.dropout_output(proj_x_l)
        proj_x_a = self.dropout_output(proj_x_a)
        proj_x_v = self.dropout_output(proj_x_v)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) 
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = h_ls.permute(1, 0, 2) # 'bt l dim'
        h_ls = self.trans_l_mem(h_ls)
        # if type(h_ls) == tuple:
        #     h_ls = h_ls[0]
        # last_h_l = last_hs = h_ls[-1]
        proj_x_l = proj_x_l.permute(1, 0, 2)
        proj_x_a = proj_x_a.permute(1, 0, 2)
        proj_x_v = proj_x_v.permute(1, 0, 2)
        l_self_att_pool = self.trans_l_self(proj_x_l)
        a_self_att_pool = self.trans_a_self(proj_x_a)
        v_self_att_pool = self.trans_v_self(proj_x_v)


        last_hs = torch.cat([h_ls, l_self_att_pool, a_self_att_pool, v_self_att_pool], dim=1)
        last_hs = self.dropout_output(last_hs)
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)

        # store results
        # emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()
        _ = []
        return features, vals_out, _ # , interloss


class MULTMOSEI_V2(nn.Module):
    def __init__(self, args):
        super(MULTMOSEI, self).__init__()

        # params: analyze args
        audio_dim   = args.input_dims[0]
        text_dim    = args.input_dims[1]
        video_dim   = args.input_dims[2]
        output_dim1 = 8 # args.output_dim1
        output_dim2 = 1 # args.output_dim1

        # params: analyze args
        self.attn_mask = True
        self.layers = 4 # args.layers # 4 
        self.dropout = args.dropout
        self.num_heads = 1 # args.num_heads # 8
        self.hidden_dim = 128 # args.hidden_dim # 128
        self.conv1d_kernel_size = 5 # args.conv1d_kernel_size # 5
        # self.grad_clip = args.grad_clip
        
        # params: intermedia
        combined_dim = 3*2*self.hidden_dim # 2 * (self.hidden_dim + self.hidden_dim + self.hidden_dim)
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(video_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_l_mem = FRA2UTT_new(2*self.hidden_dim)
        self.trans_a_mem = FRA2UTT_new(2*self.hidden_dim)
        self.trans_v_mem = FRA2UTT_new(2*self.hidden_dim)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # cls layers
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        else:
            raise ValueError("Unknown network type")
        
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, batch):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            video_feat: tensor of shape (batch, seqlen2, video_in)
            text_feat:  tensor of shape (batch, seqlen3, text_in)
        '''
        x_l = batch[0].transpose(1, 2)
        x_a = batch[1].transpose(1, 2)
        x_v = batch[2].transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) 
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = h_ls.permute(1, 0, 2)
        h_ls = self.trans_l_mem(h_ls)
        # if type(h_ls) == tuple:
        #     h_ls = h_ls[0]
        # last_h_l = last_hs = h_ls[-1]

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = h_as.permute(1, 0, 2)
        h_as = self.trans_a_mem(h_as)
        # if type(h_as) == tuple:
        #     h_as = h_as[0]
        # last_h_a = last_hs = h_as[-1]
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = h_vs.permute(1, 0, 2)
        h_vs = self.trans_v_mem(h_vs)
        # if type(h_vs) == tuple:
        #     h_vs = h_vs[0]
        # last_h_v = last_hs = h_vs[-1]
        # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        last_hs = torch.cat([h_ls, h_as, h_vs], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)

        # store results
        # emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, vals_out # , interloss


class MULTMOSE0_V1(nn.Module):
    """
    Cuda out of memory
    """
    def __init__(self, args):
        super(MULTMOSEI, self).__init__()

        # params: analyze args
        audio_dim   = args.input_dims[0]
        text_dim    = args.input_dims[1]
        video_dim   = args.input_dims[2]
        output_dim1 = 8 # args.output_dim1
        output_dim2 = 1 # args.output_dim1

        # params: analyze args
        self.attn_mask = True
        self.layers = 4 # args.layers # 4 
        self.dropout = args.dropout
        self.num_heads = 1 # args.num_heads # 8
        self.hidden_dim = 128 # args.hidden_dim # 128
        self.conv1d_kernel_size = 5 # args.conv1d_kernel_size # 5
        # self.grad_clip = args.grad_clip
        
        # params: intermedia
        combined_dim = 3*2*self.hidden_dim # 2 * (self.hidden_dim + self.hidden_dim + self.hidden_dim)
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(video_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_l_mem = FRA2UTT_new(2*self.hidden_dim)
        self.trans_a_mem = FRA2UTT_new(2*self.hidden_dim)
        self.trans_v_mem = FRA2UTT_new(2*self.hidden_dim)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # cls layers
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        else:
            raise ValueError("Unknown network type")
        
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, batch):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            video_feat: tensor of shape (batch, seqlen2, video_in)
            text_feat:  tensor of shape (batch, seqlen3, text_in)
        '''
        x_l = batch[1].transpose(1, 2)
        x_a = batch[0].transpose(1, 2)
        x_v = batch[2].transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) 
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = h_ls.permute(1, 0, 2)
        h_ls = self.trans_l_mem(h_ls)
        # if type(h_ls) == tuple:
        #     h_ls = h_ls[0]
        # last_h_l = last_hs = h_ls[-1]

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = h_as.permute(1, 0, 2)
        h_as = self.trans_a_mem(h_as)
        # if type(h_as) == tuple:
        #     h_as = h_as[0]
        # last_h_a = last_hs = h_as[-1]
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = h_vs.permute(1, 0, 2)
        h_vs = self.trans_v_mem(h_vs)
        # if type(h_vs) == tuple:
        #     h_vs = h_vs[0]
        # last_h_v = last_hs = h_vs[-1]
        # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        last_hs = torch.cat([h_ls, h_as, h_vs], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)

        # store results
        # emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()
        _ = [] 
        return features, vals_out, _ # , interloss



class MULTMOSEI_V0(nn.Module):
    def __init__(self, args):
        super(MULTMOSEI, self).__init__()

        # params: analyze args
        audio_dim   = args.input_dims[0]
        text_dim    = args.input_dims[1]
        video_dim   = args.input_dims[2]
        output_dim1 = 8 # args.output_dim1
        output_dim2 = 1 # args.output_dim1

        # params: analyze args
        self.attn_mask = True
        self.layers = 4 # args.layers # 4 
        self.dropout = args.dropout
        self.num_heads = 1 # args.num_heads # 8
        self.hidden_dim = 128 # args.hidden_dim # 128
        self.conv1d_kernel_size = 5 # args.conv1d_kernel_size # 5
        # self.grad_clip = args.grad_clip
        
        # params: intermedia
        combined_dim = 2 * (self.hidden_dim + self.hidden_dim + self.hidden_dim)
        output_dim = self.hidden_dim // 2
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(text_dim,  self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_a = nn.Conv1d(audio_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)
        self.proj_v = nn.Conv1d(video_dim, self.hidden_dim, kernel_size=self.conv1d_kernel_size, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # cls layers
        self.fc_out_1 = nn.Linear(output_dim, output_dim1)
        self.fc_out_2 = nn.Linear(output_dim, output_dim2)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.hidden_dim, self.dropout
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.hidden_dim, self.dropout
        else:
            raise ValueError("Unknown network type")
        
        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, batch):
        '''
            audio_feat: tensor of shape (batch, seqlen1, audio_in)
            video_feat: tensor of shape (batch, seqlen2, video_in)
            text_feat:  tensor of shape (batch, seqlen3, text_in)
        '''
        x_a = batch[0].transpose(1, 2)
        x_l = batch[1].transpose(1, 2)
        x_v = batch[2].transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) 
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]
        
        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        features = self.out_layer(last_hs_proj)

        # store results
        # emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        # interloss = torch.tensor(0).cuda()
        _ = []
        return features, vals_out, _