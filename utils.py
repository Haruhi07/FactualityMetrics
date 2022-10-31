import json
import pickle
import pandas as pd
from tqdm import tqdm


def pd_readfile(filedir):
    if '.pkl' in filedir:
        ret = pd.read_pickle(filedir)
    elif '.json' in filedir:
        ret = pd.read_json(filedir)
    return ret


def write_jsonl(obj, filedir):
    with open(filedir, "w") as fp:
        for idx, item in obj.iterrows():
            item_dict = item.to_dict()
            fp.write(json.dumps(item_dict) + '\n')
