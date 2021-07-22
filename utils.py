import pickle
import os
import numpy as np

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(path + " already exists.")

def sort_metadata_by_date(metadata_list, date='date_uploaded', features=None):
    if type(features) != type(None):
        if type(features) == list:
            feature_list = features
        else:
            feature_list = [features[i] for i in range(features.shape[0])]
        sort_func = lambda x : getattr(x[0], date)()
        all_meta_list_sorted = sorted(zip(metadata_list, feature_list), key=lambda x: sort_func(x))
        return all_meta_list_sorted
    else:
        sort_func = lambda meta : getattr(meta, date)()
        all_meta_list_sorted = sorted(metadata_list, key=lambda x: sort_func(x))
        return all_meta_list_sorted

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def normalize(features):
    # Normalize a 2d features
    return features / np.linalg.norm(features, axis=1).reshape(features.shape[0],1)

import more_itertools as mit
def divide(lst, n):
    """Divide successive n-sized chunks from lst."""
    return [list(c) for c in mit.divide(n, lst)]

def save_obj_as_pickle(pickle_location, obj):
    pickle.dump(obj, open(pickle_location, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Save object as a pickle at {pickle_location}")

def load_pickle(pickle_location, default_obj=None):
    if os.path.exists(pickle_location):
        try:
            return pickle.load(open(pickle_location, 'rb'))
        except ModuleNotFoundError:
            # Hack to remove a no longer existed module
            import sys, large_scale_yfcc_download
            sys.modules['flickr_parsing'] = large_scale_yfcc_download
            a = pickle.load(open(pickle_location, 'rb'))
            del sys.modules['flickr_parsing']
            save_obj_as_pickle(pickle_location, a)
            # import pdb; pdb.set_trace()
            return pickle.load(open(pickle_location, 'rb'))
    else:
        return default_obj
