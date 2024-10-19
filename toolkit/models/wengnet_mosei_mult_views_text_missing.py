import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class MatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)

class FRA2UTT(nn.Module):
    def __init__(self, input_dim=1024, atsize=1024, softmax_scale=0.3):
        super(FRA2UTT, self).__init__()
        self.atsize = atsize
        self.softmax_scale = softmax_scale
        self.attention_context_vector = nn.Parameter(torch.empty(1,atsize)) #(feature_dim)
        nn.init.xavier_normal_(self.attention_context_vector)
        self.input_proj = nn.Linear(input_dim, self.atsize)
    
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        attention_context_vector = self.attention_context_vector.repeat(batch_size,1).unsqueeze(2)
        input_proj = torch.tanh(self.input_proj(input_tensor))
        vector_attention = torch.bmm(input_proj, attention_context_vector)
        #softmax
        vector_attention = F.softmax(self.softmax_scale*vector_attention,dim=1) 
        output_vector = torch.mul(input_tensor, vector_attention)
        output_vector.squeeze() 
        output_tensor = torch.sum(output_vector, dim=1, keepdim=False)
        return output_tensor
    

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
        return output_tensor, vector_attention

class Cross_Attention(nn.Module):
    def __init__(self, input_dim=1024, atsize=1024, softmax_scale=0.3):
        super(Cross_Attention, self).__init__()
        self.atsize = atsize
        self.softmax_scale = softmax_scale
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.dropout_output = nn.Dropout(0.5)
    
    def forward(self, query_tensor, input_tensor):
        # query_tensor: batchsize*head*dim
        input_tensor = self.dropout_output(input_tensor)
        input_proj = torch.tanh(self.input_proj(input_tensor))
        batch_size = input_tensor.shape[0]

        query_tensor = self.query_proj(query_tensor)
        attention_context_vector = query_tensor.transpose(1,2)

        vector_attention = torch.bmm(input_proj, attention_context_vector)
        #softmax
        vector_attention = F.softmax(self.softmax_scale*vector_attention,dim=1)
        result_list = [torch.mul(input_tensor, vector_attention[:, :, i].unsqueeze(-1)) for i in range(vector_attention.size(2))]
        result_list = [torch.sum(tensor, dim=1, keepdim=True) for tensor in result_list]
        output_tensor = torch.cat(result_list, dim=1)
        output_tensor = self.dropout_output(output_tensor)
        return output_tensor, vector_attention

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class ResidualAE(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''
    def __init__(self, layers, n_blocks, input_dim, dropout=0.3, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim*3, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))
    
    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)
    
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer)-2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.ReLU()) # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        
        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    
    def forward(self, x_a, x_t, x_v):
        x_in = torch.cat((x_a, x_t, x_v), dim=-1)
        latents = []
        x_in = self.transition(x_in)
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            latent = encoder(x_in)
            x_out = decoder(latent) + x_t
            x_in = x_out
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return x_out


class WengnetMOSEIMultViewsTextMissing(nn.Module):
    def __init__(self, args, output_dim1=1, output_dim2=1, layers='256,128', dropout=0.3):
        super(WengnetMOSEIMultViewsTextMissing, self).__init__()
        print('init_wengnet mult modal query')
        layers_list = list(map(lambda x: int(x), layers.split(',')))
        general_dim = 256

        self.frame_dim_reshape_0 = nn.Linear(args.input_dims[0], general_dim)
        self.frame_dim_reshape_1 = nn.Linear(args.input_dims[1], general_dim)
        self.frame_dim_reshape_2 = nn.Linear(args.input_dims[2], general_dim)


        # for fused features
        fused_layer = '256,256'

        # imagination model # layers, n_blocks, input_dim, dropout=0.3, use_bn=False
        self.missing_text_imagination_mlp = ResidualAE(layers=[128], n_blocks=1, input_dim=general_dim, dropout=dropout)
        self.missing_cross_text_query_imagination_mlp = ResidualAE(layers=[64], n_blocks=1, input_dim=128, dropout=dropout)

        # for fused features
        self.fra2utt_0 = FRA2UTT_new(input_dim=general_dim)
        self.fra2utt_1 = FRA2UTT_new(input_dim=general_dim)
        self.fra2utt_2 = FRA2UTT_new(input_dim=general_dim)

        self.audio_mlp = self.MLP(general_dim, fused_layer, dropout)
        self.text_mlp  = self.MLP(general_dim, fused_layer, dropout)
        self.video_mlp = self.MLP(general_dim, fused_layer, dropout)

        
        hiddendim = general_dim * 3
        self.attention_mlp = self.MLP(hiddendim, fused_layer, dropout)
        self.fc_att   = nn.Linear(general_dim, 3)


        # for mult modal query
        self.cross_fused_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_at_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_tv_query_mlp  = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_av_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_audio_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_text_query_mlp  = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_video_query_mlp = self.MLP(general_dim, str(general_dim), dropout)

        self.cross_att_fra2utt_0 = Cross_Attention(input_dim=general_dim)
        self.cross_att_fra2utt_1 = Cross_Attention(input_dim=general_dim)
        self.cross_att_fra2utt_2 = Cross_Attention(input_dim=general_dim)

        self.cross_audio_mlp = self.MLP(general_dim, layers , dropout)
        self.cross_text_mlp  = self.MLP(general_dim, layers, dropout)
        self.cross_video_mlp = self.MLP(general_dim, layers, dropout)

        hiddendim = layers_list[-1] * 7
        self.cross_attention_mlp = self.MLP(hiddendim, layers, dropout)
        self.cross_fc_att   = nn.Linear(layers_list[-1], 7)


        self.fc_out_e = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim2)
        self.fc_out_ev = nn.Linear(output_dim1, output_dim2)

        self.orgin_linear_change = nn.Sequential(
            nn.Linear(layers_list[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.dropout_output = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.matmul = MatMul()
        self.prelu = nn.PReLU(num_parameters=6, init=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.normali = NormalizeLayer()
        self.layer_normali = torch.nn.LayerNorm(general_dim)
        
        # self.fc_out_att_ev = nn.Linear(2, 1)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, batch):  # 输入缺失文本模态的流
        audio_feat, text_feat, video_feat = batch[0], batch[1], batch[2]

        missing_flag = batch[-1]
        # cross_fused_feat, feat4rnc, multi_query, cross_hiddens = full_rep[0], full_rep[1], full_rep[2], full_rep[3]

        # input reshape
        audio_feat = self.frame_dim_reshape_0(audio_feat)
        text_feat = self.frame_dim_reshape_1(text_feat)
        video_feat = self.frame_dim_reshape_2(video_feat)


        # mult fram to fused_feat
        audio_hidden_pre, attention_0 = self.fra2utt_0(audio_feat)
        text_hidden_pre, attention_1 = self.fra2utt_1(text_feat)
        video_hidden_pre, attention_2 = self.fra2utt_2(video_feat)
        attention_masks = [attention_0, attention_1, attention_2]

        audio_hidden = self.audio_mlp(audio_hidden_pre) # [32, 128]
        text_hidden  = self.text_mlp(text_hidden_pre)   # [32, 128]
        video_hidden = self.video_mlp(video_hidden_pre) # [32, 128]

        #************image 预测新的text_hidden，替换这里原始的text_hidden**************
        # if missing_flag:
        #     text_hidden = self.missing_text_imagination_mlp(audio_hidden, text_hidden, video_hidden)

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]
        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]

        fused_feat = self.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        # fused_feat = self.normali(fused_feat)

        fused_feat_at = self.matmul(multi_hidden2[:, :, :2], attention[:, :2, :])
        fused_feat_at = fused_feat_at.squeeze() # [32, 128]

        fused_feat_tv = self.matmul(multi_hidden2[:, :, 1:], attention[:, 1:, :])
        fused_feat_tv = fused_feat_tv.squeeze() # [32, 128]

        multi_hidden2_av = torch.stack((multi_hidden2[:, :, 0], multi_hidden2[:, :, 2]), dim=2)
        attention_av = torch.stack((attention[:, 0, :], attention[:, 2, :]), dim=1)
        fused_feat_av = self.matmul(multi_hidden2_av, attention_av)
        fused_feat_av = fused_feat_av.squeeze() # [32, 128]

        
        # mult modal query 
        fused_feat = self.cross_fused_query_mlp(fused_feat)
        at_hidden = self.cross_at_query_mlp(fused_feat_at)
        tv_hidden = self.cross_tv_query_mlp(fused_feat_tv)
        av_hidden = self.cross_av_query_mlp(fused_feat_av)
        audio_hidden = self.cross_audio_query_mlp(audio_hidden)
        text_hidden = self.cross_text_query_mlp(text_hidden)
        video_hidden = self.cross_video_query_mlp(video_hidden)

        multi_query = torch.stack([fused_feat, at_hidden, tv_hidden, av_hidden,  audio_hidden, text_hidden, video_hidden], dim=1)

        cross_audio_hidden, _ = self.cross_att_fra2utt_0(multi_query, audio_feat)
        cross_text_hidden, _ = self.cross_att_fra2utt_1(multi_query, text_feat)
        cross_video_hidden, _ = self.cross_att_fra2utt_2(multi_query, video_feat)

        cross_audio_hidden = self.cross_audio_mlp(cross_audio_hidden)
        cross_text_hidden = self.cross_text_mlp(cross_text_hidden)
        cross_video_hidden = self.cross_video_mlp(cross_video_hidden)
    
        #**************image 预测新的cross_text_hidden，替换这里的cross_text_hidden**************
        # if missing_flag:
        #     cross_text_hidden = self.missing_cross_text_query_imagination_mlp(cross_audio_hidden, cross_text_hidden, cross_video_hidden)

        cross_hiddens = torch.stack([cross_audio_hidden, cross_text_hidden, cross_video_hidden], dim=1)
        attention_tensor = torch.unsqueeze(attention, 3)
        weighted_cross_hiddens = attention_tensor*cross_hiddens
        weighted_cross_hiddens = weighted_cross_hiddens.sum(dim=1)


        cross_multi_hidden1 =weighted_cross_hiddens.view(weighted_cross_hiddens.shape[0], weighted_cross_hiddens.shape[1]*weighted_cross_hiddens.shape[2]) # [32, 384]
        cross_attention = self.cross_attention_mlp(cross_multi_hidden1)
        cross_attention = self.cross_fc_att(cross_attention)
        cross_attention = torch.unsqueeze(cross_attention, 2) # [32, 4, 1]
        cross_multi_hidden2 = weighted_cross_hiddens.transpose(1,2) # [32, 128, 4]
        cross_fused_feat = self.matmul(cross_multi_hidden2, cross_attention)
        cross_fused_feat = cross_fused_feat.squeeze() # [32, 128]
        # cross_fused_feat = self.normali(cross_fused_feat)

        # emos_out  = self.fc_out_e(fused_feat)
        # vals_out_e  = self.fc_out_ev(emos_out)
        # vals_out_e = self.tanh(vals_out_e)
        vals_out_v = self.fc_out_v(cross_fused_feat)
        # vals_out_total = torch.cat([vals_out_e, vals_out_v], dim=1)
        vals_out = vals_out_v

        feat4rnc = self.orgin_linear_change(cross_fused_feat)

        return vals_out, [cross_fused_feat, feat4rnc, text_hidden, cross_hiddens[:,1]]

class WengnetMOSEI_0_4962(nn.Module):
    def __init__(self, args, output_dim1=1, output_dim2=1, layers='256,128', dropout=0.3):
        super(WengnetMOSEI, self).__init__()
        print('init_wengnet mult modal query')
        layers_list = list(map(lambda x: int(x), layers.split(',')))
        general_dim = 256

        self.frame_dim_reshape_0 = nn.Linear(args.input_dims[0], general_dim)
        self.frame_dim_reshape_1 = nn.Linear(args.input_dims[1], general_dim)
        self.frame_dim_reshape_2 = nn.Linear(args.input_dims[2], general_dim)


        # for fused features
        fused_layer = '256,256'

        self.fra2utt_0 = FRA2UTT_new(input_dim=general_dim)
        self.fra2utt_1 = FRA2UTT_new(input_dim=general_dim)
        self.fra2utt_2 = FRA2UTT_new(input_dim=general_dim)

        self.audio_mlp = self.MLP(general_dim, fused_layer, dropout)
        self.text_mlp  = self.MLP(general_dim, fused_layer, dropout)
        self.video_mlp = self.MLP(general_dim, fused_layer, dropout)

        
        hiddendim = general_dim * 3
        self.attention_mlp = self.MLP(hiddendim, fused_layer, dropout)
        self.fc_att   = nn.Linear(general_dim, 3)


        # for mult modal query
        self.cross_fused_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_at_query_mlp = self.MLP(general_dim, str(general_dim) , dropout)
        self.cross_tv_query_mlp  = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_av_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_audio_query_mlp = self.MLP(general_dim, str(general_dim) , dropout)
        self.cross_text_query_mlp  = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_video_query_mlp = self.MLP(general_dim, str(general_dim), dropout)

        self.cross_att_fra2utt_0 = Cross_Attention(input_dim=general_dim)
        self.cross_att_fra2utt_1 = Cross_Attention(input_dim=general_dim)
        self.cross_att_fra2utt_2 = Cross_Attention(input_dim=general_dim)

        self.cross_audio_mlp = self.MLP(general_dim, layers , dropout)
        self.cross_text_mlp  = self.MLP(general_dim, layers, dropout)
        self.cross_video_mlp = self.MLP(general_dim, layers, dropout)

        hiddendim = layers_list[-1] * 7
        self.cross_attention_mlp = self.MLP(hiddendim, layers, dropout)
        self.cross_fc_att   = nn.Linear(layers_list[-1], 7)


        self.fc_out_e = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim2)
        self.fc_out_ev = nn.Linear(output_dim1, output_dim2)

        self.dropout_output = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.matmul = MatMul()
        self.prelu = nn.PReLU(num_parameters=6, init=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.normali = NormalizeLayer()
        
        # self.fc_out_att_ev = nn.Linear(2, 1)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, batch):
        audio_feat, text_feat, video_feat = batch[0], batch[1], batch[2]


        # input reshape
        audio_feat = self.frame_dim_reshape_0(audio_feat)
        text_feat = self.frame_dim_reshape_1(text_feat)
        video_feat = self.frame_dim_reshape_2(video_feat)


        # mult fram to fused_feat
        audio_hidden_pre, attention_0 = self.fra2utt_0(audio_feat)
        text_hidden_pre, attention_1 = self.fra2utt_1(text_feat)
        video_hidden_pre, attention_2 = self.fra2utt_2(video_feat)
        attention_masks = [attention_0, attention_1, attention_2]

        audio_hidden = self.audio_mlp(audio_hidden_pre) # [32, 128]
        text_hidden  = self.text_mlp(text_hidden_pre)   # [32, 128]
        video_hidden = self.video_mlp(video_hidden_pre) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]
        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]

        fused_feat = self.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        # fused_feat = self.normali(fused_feat)

        fused_feat_at = self.matmul(multi_hidden2[:, :, :2], attention[:, :2, :])
        fused_feat_at = fused_feat_at.squeeze() # [32, 128]

        fused_feat_tv = self.matmul(multi_hidden2[:, :, 1:], attention[:, 1:, :])
        fused_feat_tv = fused_feat_tv.squeeze() # [32, 128]

        multi_hidden2_av = torch.stack((multi_hidden2[:, :, 0], multi_hidden2[:, :, 2]), dim=2)
        attention_av = torch.stack((attention[:, 0, :], attention[:, 2, :]), dim=1)
        fused_feat_av = self.matmul(multi_hidden2_av, attention_av)
        fused_feat_av = fused_feat_av.squeeze() # [32, 128]

        
        # mult modal query 
        fused_feat = self.cross_fused_query_mlp(fused_feat)
        at_hidden = self.cross_at_query_mlp(fused_feat_at)
        tv_hidden = self.cross_tv_query_mlp(fused_feat_tv)
        av_hidden = self.cross_av_query_mlp(fused_feat_av)
        audio_hidden = self.cross_audio_query_mlp(audio_hidden)
        text_hidden = self.cross_text_query_mlp(text_hidden)
        video_hidden = self.cross_video_query_mlp(video_hidden)

        multi_query = torch.stack([fused_feat, at_hidden, tv_hidden, av_hidden,  audio_hidden, text_hidden, video_hidden], dim=1)

        cross_audio_hidden, _ = self.cross_att_fra2utt_0(multi_query, audio_feat)
        cross_text_hidden, _ = self.cross_att_fra2utt_1(multi_query, text_feat)
        cross_video_hidden, _ = self.cross_att_fra2utt_2(multi_query, video_feat)

        cross_audio_hidden = self.cross_audio_mlp(cross_audio_hidden)
        cross_text_hidden = self.cross_text_mlp(cross_text_hidden)
        cross_video_hidden = self.cross_video_mlp(cross_video_hidden)

        cross_hiddens = torch.stack([cross_audio_hidden, cross_text_hidden, cross_video_hidden], dim=1)
        attention_tensor = torch.unsqueeze(attention, 3)
        weighted_cross_hiddens = attention_tensor*cross_hiddens
        weighted_cross_hiddens = weighted_cross_hiddens.sum(dim=1)


        cross_multi_hidden1 =weighted_cross_hiddens.view(weighted_cross_hiddens.shape[0], weighted_cross_hiddens.shape[1]*weighted_cross_hiddens.shape[2]) # [32, 384]
        cross_attention = self.cross_attention_mlp(cross_multi_hidden1)
        cross_attention = self.cross_fc_att(cross_attention)
        cross_attention = torch.unsqueeze(cross_attention, 2) # [32, 4, 1]
        cross_multi_hidden2 = weighted_cross_hiddens.transpose(1,2) # [32, 128, 4]
        cross_fused_feat = self.matmul(cross_multi_hidden2, cross_attention)
        cross_fused_feat = cross_fused_feat.squeeze() # [32, 128]
        # cross_fused_feat = self.normali(cross_fused_feat)

        # emos_out  = self.fc_out_e(fused_feat)
        # vals_out_e  = self.fc_out_ev(emos_out)
        # vals_out_e = self.tanh(vals_out_e)
        vals_out_v = self.fc_out_v(cross_fused_feat)
        # vals_out_total = torch.cat([vals_out_e, vals_out_v], dim=1)
        vals_out = vals_out_v
        return cross_fused_feat, vals_out, attention_masks


class WengnetMOSEI_0_5019(nn.Module):
    def __init__(self, args, output_dim1=1, output_dim2=1, layers='256,128', dropout=0.3):
        super(WengnetMOSEI, self).__init__()
        print('init_wengnet mult modal query')
        layers_list = list(map(lambda x: int(x), layers.split(',')))
        general_dim = 256

        self.frame_dim_reshape_0 = nn.Linear(args.input_dims[0], general_dim)
        self.frame_dim_reshape_1 = nn.Linear(args.input_dims[1], general_dim)
        self.frame_dim_reshape_2 = nn.Linear(args.input_dims[2], general_dim)


        # for fused features
        fused_layer = '256,256'

        self.fra2utt_0 = FRA2UTT_new(input_dim=general_dim)
        self.fra2utt_1 = FRA2UTT_new(input_dim=general_dim)
        self.fra2utt_2 = FRA2UTT_new(input_dim=general_dim)

        self.audio_mlp = self.MLP(general_dim, fused_layer, dropout)
        self.text_mlp  = self.MLP(general_dim, fused_layer, dropout)
        self.video_mlp = self.MLP(general_dim, fused_layer, dropout)

        
        hiddendim = general_dim * 3
        self.attention_mlp = self.MLP(hiddendim, fused_layer, dropout)
        self.fc_att   = nn.Linear(general_dim, 3)


        # for mult modal query
        self.cross_fused_query_mlp = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_audio_query_mlp = self.MLP(general_dim, str(general_dim) , dropout)
        self.cross_text_query_mlp  = self.MLP(general_dim, str(general_dim), dropout)
        self.cross_video_query_mlp = self.MLP(general_dim, str(general_dim), dropout)

        self.cross_att_fra2utt_0 = Cross_Attention(input_dim=general_dim)
        self.cross_att_fra2utt_1 = Cross_Attention(input_dim=general_dim)
        self.cross_att_fra2utt_2 = Cross_Attention(input_dim=general_dim)

        self.cross_audio_mlp = self.MLP(general_dim, layers , dropout)
        self.cross_text_mlp  = self.MLP(general_dim, layers, dropout)
        self.cross_video_mlp = self.MLP(general_dim, layers, dropout)

        hiddendim = layers_list[-1] * 4
        self.cross_attention_mlp = self.MLP(hiddendim, layers, dropout)
        self.cross_fc_att   = nn.Linear(layers_list[-1], 4)


        self.fc_out_e = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim2)
        self.fc_out_ev = nn.Linear(output_dim1, output_dim2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.matmul = MatMul()
        self.prelu = nn.PReLU(num_parameters=6, init=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.normali = NormalizeLayer()
        
        # self.fc_out_att_ev = nn.Linear(2, 1)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, batch):
        audio_feat, text_feat, video_feat = batch[0], batch[1], batch[2]


        # input reshape
        audio_feat = self.frame_dim_reshape_0(audio_feat)
        text_feat = self.frame_dim_reshape_1(text_feat)
        video_feat = self.frame_dim_reshape_2(video_feat)


        # mult fram to fused_feat
        audio_hidden_pre, attention_0 = self.fra2utt_0(audio_feat)
        text_hidden_pre, attention_1 = self.fra2utt_1(text_feat)
        video_hidden_pre, attention_2 = self.fra2utt_2(video_feat)
        attention_masks = [attention_0, attention_1, attention_2]

        audio_hidden = self.audio_mlp(audio_hidden_pre) # [32, 128]
        text_hidden  = self.text_mlp(text_hidden_pre)   # [32, 128]
        video_hidden = self.video_mlp(video_hidden_pre) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]
        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = self.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        fused_feat = self.normali(fused_feat)


        # mult modal query 
        fused_feat = self.cross_fused_query_mlp(fused_feat)
        audio_hidden = self.cross_fused_query_mlp(audio_hidden)
        text_hidden = self.cross_fused_query_mlp(text_hidden)
        video_hidden = self.cross_fused_query_mlp(video_hidden)



        multi_query = torch.stack([fused_feat, audio_hidden, text_hidden, video_hidden], dim=1)
        cross_audio_hidden, _ = self.cross_att_fra2utt_0(multi_query, audio_feat)
        cross_text_hidden, _ = self.cross_att_fra2utt_1(multi_query, text_feat)
        cross_video_hidden, _ = self.cross_att_fra2utt_2(multi_query, video_feat)

        cross_audio_hidden = self.cross_audio_mlp(cross_audio_hidden)
        cross_text_hidden = self.cross_text_mlp(cross_text_hidden)
        cross_video_hidden = self.cross_video_mlp(cross_video_hidden)

        cross_hiddens = torch.stack([cross_audio_hidden, cross_text_hidden, cross_video_hidden], dim=1)
        attention_tensor = torch.unsqueeze(attention, 3)
        weighted_cross_hiddens = attention_tensor*cross_hiddens
        weighted_cross_hiddens = weighted_cross_hiddens.sum(dim=1)


        cross_multi_hidden1 =weighted_cross_hiddens.view(weighted_cross_hiddens.shape[0], weighted_cross_hiddens.shape[1]*weighted_cross_hiddens.shape[2]) # [32, 384]
        cross_attention = self.cross_attention_mlp(cross_multi_hidden1)
        cross_attention = self.cross_fc_att(cross_attention)
        cross_attention = torch.unsqueeze(cross_attention, 2) # [32, 4, 1]
        cross_multi_hidden2 = weighted_cross_hiddens.transpose(1,2) # [32, 128, 4]
        cross_fused_feat = self.matmul(cross_multi_hidden2, cross_attention)
        cross_fused_feat = cross_fused_feat.squeeze() # [32, 128]
        # cross_fused_feat = self.normali(cross_fused_feat)

        # emos_out  = self.fc_out_e(fused_feat)
        # vals_out_e  = self.fc_out_ev(emos_out)
        # vals_out_e = self.tanh(vals_out_e)
        vals_out_v = self.fc_out_v(cross_fused_feat)
        # vals_out_total = torch.cat([vals_out_e, vals_out_v], dim=1)
        vals_out = vals_out_v
        return fused_feat, vals_out, attention_masks

class WengnetMOSEI_classic(nn.Module):
    def __init__(self, args, output_dim1=1, output_dim2=1, layers='256,128', dropout=0.3):
        super(WengnetMOSEI, self).__init__()
        print('init_wengnet classic')
        self.fra2utt_0 = FRA2UTT_new(input_dim=args.input_dims[0])
        self.fra2utt_1 = FRA2UTT_new(input_dim=args.input_dims[1])
        self.fra2utt_2 = FRA2UTT_new(input_dim=args.input_dims[2])

        self.audio_mlp = self.MLP(args.input_dims[0], layers, dropout)
        self.text_mlp  = self.MLP(args.input_dims[1], layers, dropout)
        self.video_mlp = self.MLP(args.input_dims[2], layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_e = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim2)
        self.fc_out_ev = nn.Linear(output_dim1, output_dim2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.matmul = MatMul()
        self.prelu = nn.PReLU(num_parameters=6, init=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.normali = NormalizeLayer()
        
        # self.fc_out_att_ev = nn.Linear(2, 1)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, batch):
        audio_feat, text_feat, video_feat = batch[0], batch[1], batch[2]

        audio_feat, attention_0 = self.fra2utt_0(audio_feat)
        text_feat, attention_1 = self.fra2utt_1(text_feat)
        video_feat, attention_2 = self.fra2utt_2(video_feat)
        attention_masks = [attention_0, attention_1, attention_2]
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128]
        video_hidden = self.video_mlp(video_feat) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = self.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        # fused_feat = self.normali(fused_feat)

        # emos_out  = self.fc_out_e(fused_feat)
        # vals_out_e  = self.fc_out_ev(emos_out)
        # vals_out_e = self.tanh(vals_out_e)
        vals_out_v = self.fc_out_v(fused_feat)
        # vals_out_total = torch.cat([vals_out_e, vals_out_v], dim=1)
        vals_out = vals_out_v
        return fused_feat, vals_out, attention_masks


class WengnetMOSEI_PRETRAIN(nn.Module):
    def __init__(self, args, output_dim1=1, output_dim2=1, layers='256,128', dropout=0.3):
        super(WengnetMOSEI, self).__init__()
        print('init_wengnet')
        self.fra2utt_0 = FRA2UTT_new(input_dim=args.input_dims[0])
        self.fra2utt_1 = FRA2UTT_new(input_dim=args.input_dims[1])
        self.fra2utt_2 = FRA2UTT_new(input_dim=args.input_dims[2])

        self.audio_mlp = self.MLP(args.input_dims[0], layers, dropout)
        self.text_mlp  = self.MLP(args.input_dims[1], layers, dropout)
        self.video_mlp = self.MLP(args.input_dims[2], layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_e = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim2)
        self.fc_out_ev = nn.Linear(output_dim1, output_dim2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.matmul = MatMul()
        self.prelu = nn.PReLU(num_parameters=6, init=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.normali = NormalizeLayer()
        
        # self.fc_out_att_ev = nn.Linear(2, 1)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, batch):
        audio_feat, text_feat, video_feat = batch[0], batch[1], batch[2]
        # audio_feat = self.softmax(audio_feat)
        # text_feat = self.softmax(text_feat)
        # video_feat = self.softmax(video_feat)
        audio_feat, attention_0 = self.fra2utt_0(audio_feat)
        text_feat, attention_1 = self.fra2utt_1(text_feat)
        video_feat, attention_2 = self.fra2utt_2(video_feat)
        attention_masks = [attention_0, attention_1, attention_2]
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128]
        video_hidden = self.video_mlp(video_feat) # [32, 128]
        # print(audio_hidden)
        # print(text_hidden)
        # print(video_hidden)

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = self.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        fused_feat = self.normali(fused_feat)
        # emos_out  = self.fc_out_e(fused_feat)
        # vals_out_e  = self.fc_out_ev(emos_out)
        # vals_out_e = self.tanh(vals_out_e)
        # vals_out_v = self.fc_out_v(fused_feat)
        # vals_out_total = torch.cat([vals_out_e, vals_out_v], dim=1)
        # vals_out = vals_out_v
        return fused_feat, attention_masks