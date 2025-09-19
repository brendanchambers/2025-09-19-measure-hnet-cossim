#  https://github.com/main-horse/hnet-impl

import torch
from hnet_impl import HNetLM, HNetConfig, completion_sync
from hnet_impl.torchisms import rand_njt_iids

print('successfully imported hnet_impl!')

c = HNetConfig.create_reasonable_config(D=[512,1024], arch=['m4','T9'])
with torch.device('cuda'): m = HNetLM(c).bfloat16()

print('minimal inference example...')

# inference
iids = rand_njt_iids(docs=16, slen=range(128,1024)).cuda()
logits,_ = m(iids)

print('minimal training example...')

# training
lbls = iids.long() # i.e. torch.randint_like(iids)
(celoss,_),extra = m(iids,lbls)