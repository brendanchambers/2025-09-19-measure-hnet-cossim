
## preliminaries  

`bash scripts/setup/first_time_setup.sh`  

## sanity check

`pushd hnet-impl && uv run python ../src/examples/usage.py && popd`  
Running this from the hnet-impl dir for now, to avoid refactoring the python imports.  
✓  

`pushd hnet-impl && uv run -m hnet_impl.norm && popd`  
seeing warning but looks ok for now: <frozen runpy>:128: RuntimeWarning: 'hnet_impl.norm' found in sys.modules after import of package 'hnet_impl', but prior to execution of 'hnet_impl.norm'; this may result in unpredictable behaviour
✓

`pushd hnet-impl && uv run -m hnet_impl.lin && popd`  
seeing warning but looks ok for now: <frozen runpy>:128: RuntimeWarning: 'hnet_impl.lin' found in sys.modules after import of package 'hnet_impl', but prior to execution of 'hnet_impl.lin'; this may result in unpredictable behaviour  
✓  

# NOTE: download hnet_2stage_XL from somewhere first.
# cp /path/to/hnet/hnet_2stage_XL.pt /path/to/hnet/config/hnet_2stage_XL.json .
uv run -m hnet_impl.modeling_hnet

## inference hnet model  


