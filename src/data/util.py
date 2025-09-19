

import json

def load_jsonl_to_list(path):
    data_obj_list = []
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            data_obj_list.append(obj)
    return data_obj_list