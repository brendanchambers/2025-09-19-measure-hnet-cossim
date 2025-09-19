from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from hnet_impl.torchisms import torch, TT, nn, F, nested, NJT, summon_full_params
from hnet_impl.conceptual import BlockBoundaryMixin, get_seq_idx
from hnet_impl.config_hnet import HNetConfig
from hnet_impl.xf import Isotropic
from hnet_impl.lin import Lin, HighPrecLinear, LMHead

from hnet_impl.modeling_hnet import HNetLM, test_fwd_correctness

import sys
from pathlib import Path
print(f"initial sys.path: \n{sys.path}\n")
# Add the parent directory of your package to sys.path (tmp hack to avoid refactor)
sys.path.append(str(Path(__file__).parent.parent))  # points to "src" of main project
print(f"updated sys.path: \n{sys.path}\n")
from data.util import load_jsonl_to_list



## config here for now:

sample_data_path = "/home/brendanchambers/2025-09-19-measure-hnet-cossim/data/sample/quick.jsonl"
text_key = 'text'


def run_example():

    print('fetching dataset...')
    dataset = load_jsonl_to_list(sample_data_path)
    print(dataset[0])

    print('loading model...')
    from hnet_impl.sampling import ByteTokenizer
    import re
    ## load hardcoded model
    c = HNetConfig.load_config("hnet_2stage_XL.json")
    t = ByteTokenizer()
    with torch.device("cuda"):
        m = HNetLM(c).bfloat16()
    m.load_goomba_ckpt("hnet_2stage_XL.pt")


    print('running inference examples...')
    from hnet_impl.sampling import completion_sync
    ## check greedy sampling result
    torch.manual_seed(0)

    for entry in dataset:
        text = entry[text_key]
        print(text)
        print('>>>>>>>>>>>>>>>')
        comp = completion_sync(text, t, m, max_new=200, temp=0.0001, min_p=0.0001)
        print('with p(boundary) highlighting:')
        print(comp)
        comp = re.sub(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]", "", comp)
        print('stripped:')
        print(comp)
        print('--------------')


if __name__ == "__main__":
    run_example()
