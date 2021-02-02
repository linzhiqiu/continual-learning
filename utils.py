import pickle
import os
import numpy as np

def sort_metadata_by_date(metadata_list, date='date_uploaded'):
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
    pickle.dump(obj, open(pickle_location, 'wb+'))
    print(f"Save object as a pickle at {pickle_location}")

def load_pickle(pickle_location, default_obj=None):
    if os.path.exists(pickle_location):
        return pickle.load(open(pickle_location, 'rb'))
    else:
        return default_obj