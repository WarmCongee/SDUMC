"""
get_models: get models and load default configs; 
link: https://github.com/thuiar/MMSA-FET/tree/master
"""
import torch

from .tfn import TFN
from .lmf import LMF
from .mfn import MFN
from .mfm import MFM
from .mult import MULT
from .misa import MISA
from .mctn import MCTN
from .mmim import MMIM
from .graph_mfn import Graph_MFN
from .attention import Attention
from .wengnet_mosei import WengnetMOSEI
from .wengnet_mosei_mult_views import WengnetMOSEIMultViews
from .wengnet_mosei_mult_views_text_missing import WengnetMOSEIMultViewsTextMissing
from .wengnet_mosei_mviews_llm_decode_wav import WengnetMOSEIMultViewsVicuna
from .wengnet_mosei_feat4 import WengnetMOSEI_FEAT4
from .wengnet_mosei_feat4_fine import WengnetMOSEI_FEAT4_FINE
from .wengnet_mosei_feat4_emoval import WengnetMOSEI_FEAT4_EMO2VAL
from .wengnet_mosei_emoval import WengnetMOSEI_EMOVAL
from .mult_mosei import MULTMOSEI
from .wengnet_mer2023 import WengnetMER2023
from .dst_att import DST_ATT

class get_models(torch.nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高
        args.dim = 1024

        MODEL_MAP = {
            
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            'attention': Attention,
            'lmf': LMF,
            'misa': MISA,
            'mmim': MMIM,
            'tfn': TFN,
            
            # 只支持align
            'mfn': MFN, # slow
            'graph_mfn': Graph_MFN, # slow
            'mfm': MFM, # slow
            'mctn': MCTN, # slow

            # 支持align/unalign
            'mult': MULT, # slow
            'mult_mosei': MULTMOSEI,
            'wengnet_mosei': WengnetMOSEI,
            'wengnet_mosei_mult_views': WengnetMOSEIMultViews,
            'wengnet_mosei_mult_views_text_missing': WengnetMOSEIMultViewsTextMissing,
            'wengnet_mosei_mviews_llm_decode_wav': WengnetMOSEIMultViewsVicuna,
            'wengnet_mosei_feat4_fine': WengnetMOSEI_FEAT4_FINE,
            'wengnet_mosei_feat4': WengnetMOSEI_FEAT4,
            'wengnet_mosei_feat4_emoval':WengnetMOSEI_FEAT4_EMO2VAL,
            'wengnet_mosei_emoval': WengnetMOSEI_EMOVAL,
            'dst_att': DST_ATT,
            'wengnet_mer2023': WengnetMER2023,


        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
