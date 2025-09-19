#  https://github.com/main-horse/hnet-impl

import torch
from hnet_impl import HNetLM, HNetConfig, completion_sync
from hnet_impl.torchisms import rand_njt_iids

c = HNetConfig.create_reasonable_config(D=[512,1024], arch=['m4','T9'])
with torch.device('cuda'): m = HNetLM(c).bfloat16()

# inference
iids = rand_njt_iids(docs=16, slen=range(128,1024)).cuda()
logits,_ = m(iids)

# training
lbls = iids.long() # i.e. torch.randint_like(iids)
(celoss,_),extra = m(iids,lbls)