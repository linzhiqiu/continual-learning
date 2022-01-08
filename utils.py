import pickle
import os
import numpy as np
import json

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path + " already exists.")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def normalize(features):
    # Normalize a matrix of 2d features
    return features / np.linalg.norm(features, axis=1).reshape(features.shape[0],1)

import more_itertools as mit
def divide(lst, n):
    """Divide successive n-sized chunks from lst."""
    return [list(c) for c in mit.divide(n, lst)]

def save_as_json(json_location, obj):
    with open(json_location, "w+") as f:
        json.dump(obj, f)

def load_json(json_location, default_obj=None):
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                # import pdb; pdb.set_trace()
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj

def save_obj_as_pickle(pickle_location, obj):
    pickle.dump(obj, open(pickle_location, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Save object as a pickle at {pickle_location}")

def load_pickle(pickle_location, default_obj=None):
    if os.path.exists(pickle_location):
        # return pickle.load(open(pickle_location, 'rb'))
        try:
            return pickle.load(open(pickle_location, 'rb'))
        except ModuleNotFoundError:
            import sys, temp
            # Hack to rename a module (from large_scale_yfcc_download to yfcc_download)
            sys.modules['large_scale_yfcc_download_parallel'] = temp
            sys.modules['large_scale_yfcc_download'] = temp
            # sys.modules['temp'] = yfcc_download
            a = pickle.load(open(pickle_location, 'rb'))
            save_obj_as_pickle(pickle_location, a)
            return pickle.load(open(pickle_location, 'rb'))

            # Hack to remove a no longer existed module (e.g., remove flickr_parsing)
            # sys.modules['flickr_parsing'] = yfcc_download
            # a = pickle.load(open(pickle_location, 'rb'))
            # del sys.modules['flickr_parsing']
            # save_obj_as_pickle(pickle_location, a)
            # return pickle.load(open(pickle_location, 'rb'))
    else:
        return default_obj
