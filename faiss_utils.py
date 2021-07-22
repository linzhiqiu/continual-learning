import sys
import os
sys.path.append("./CLIP")
import clip
import torch
from PIL import Image
import numpy as np
import time

from utils import load_pickle, normalize
from large_scale_yfcc_download import FlickrAccessor
import faiss

MAX_SIZE = 1000000
# MAX_SIZE = 100000

def _get_flickr_folder(folder_path):
    return os.path.join(folder_path, "all_folders.pickle")

def _get_feature_name(folder_path, model_name, normalize=True):
    if normalize:
        normalize_str = "_normalized"
    else:
        normalize_str = ""
    return os.path.join(folder_path, f"features_{model_name.replace(os.sep, '_')}{normalize_str}.pickle")

def _path_iterator_for_numpy(paths, mapping):
    # mapping is the selected indices to return
    cur_count = 0
    for path in paths:
        matrix = load_pickle(path)
        new_count = cur_count + matrix.shape[0]
        cur_mapping = list(filter(lambda x : x < new_count and x >= cur_count, mapping))
        relative_cur_mapping = [i - cur_count for i in cur_mapping]
        curr_matrix = matrix[relative_cur_mapping]
        cur_count = new_count
        yield curr_matrix

def aggregate_for_numpy(paths, mapping):
    """Return a numpy matrix of selected features indicated by mapping
        Args:
            paths (list of path to saved numpy matrix)
            mapping (indices to select)
        Returns:
            total_matrix (a single matrix with size (len(mapping) x feature_dim))
    """
    total_matrix = None
    cur_count = 0
    for path in paths:
        matrix = load_pickle(path)
        new_count = cur_count + matrix.shape[0]
        cur_mapping = list(filter(lambda x : x < new_count and x >= cur_count, mapping))
        relative_cur_mapping = [i - cur_count for i in cur_mapping]
        curr_matrix = matrix[relative_cur_mapping]
        if type(total_matrix) == type(None):
            total_matrix = curr_matrix
        else:
            total_matrix = np.concatenate((total_matrix, curr_matrix), axis=0)
        cur_count = new_count
    return total_matrix

def aggregate_for_lists(lists, mapping):
    """Return a list of selected features indicated by mapping
        Args:
            paths (list of features)
            mapping (indices to select)
        Returns:
            total_list (a single list with size (len(mapping)))
    """
    total_list = []
    cur_count = 0
    for lst in lists:
        new_count = cur_count + len(lst)
        cur_mapping = list(filter(lambda x : x < new_count and x >= cur_count, mapping))
        relative_cur_mapping = [i - cur_count for i in cur_mapping]
        for i in relative_cur_mapping:
            total_list.append(lst[i])
        cur_count = new_count
    return total_list

def _get_total_feature_length(paths):
    """Returns the total size of all features (distributed in multiple paths)
    """
    feature_count = 0
    for path in paths:
        matrix = load_pickle(path)
        feature_count += matrix.shape[0]
    return feature_count
    
def _matrix_iterator(matrix, chunk_size):
    """Generator for matrices (splitted to [chunk_size] chunks)
    """
    for i0 in range(0, matrix.shape[0], chunk_size):
        yield matrix[i0:i0 + chunk_size]

def _chunk_iterator(length, chunk_size):
    """Generator for lists ([0,...,length] splitted to [chunk_size] chunks)
    """
    lst = list(range(length))
    for i0 in range(0, length, chunk_size):
        yield lst[i0:i0 + chunk_size]

def _remove_multiple_indices_at_once(lst, removed_indices):
    mapped_removed_indices = [lst[j] for j in removed_indices]
    
    for ele in sorted(removed_indices, reverse=True):  
        del lst[ele]
    return lst, mapped_removed_indices

class _ResultHeap:
    """Accumulate query results from a sliced dataset. The final result will
    be in self.D, self.I."""

    def __init__(self, nq, k):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        heaps = faiss.float_minheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(self.D)
        heaps.ids = faiss.swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_result(self, D, I):
        """D, I do not need to be in a particular order (heap or sorted)"""
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        self.heaps.addn_with_ids(
            self.k, faiss.swig_ptr(D),
            faiss.swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()

def knn_ground_truth(xq, db_iterator, k):
    """Computes the exact KNN search results for a dataset that possibly
    does not fit in RAM but for whihch we have an iterator that
    returns it block by block.
    """
    nq, d = xq.shape
    rh = _ResultHeap(nq, k)

    index = faiss.IndexFlatIP(d)
    if faiss.get_num_gpus():
        index = faiss.index_cpu_to_all_gpus(index)

    # compute ground-truth by blocks of bs, and add to heaps
    i0 = 0
    for xbi in db_iterator:
        ni = xbi.shape[0]
        index.add(xbi)
        D, I = index.search(xq, k)
        I += i0
        rh.add_result(D, I)
        index.reset()
        i0 += ni

    rh.finalize()

    return rh.D, rh.I


class KNearestFaissFeatureChunks():
    def __init__(self, clip_features_normalized_paths, model, preprocess):
        self.clip_features_normalized_paths = clip_features_normalized_paths
        self.total_feature_num = _get_total_feature_length(self.clip_features_normalized_paths)
        print(f"Current chunk has {self.total_feature_num} features")
        self.model = model
        self.preprocess = preprocess
    
    def grab_bottom_query_indices(self, query, start_idx=0, end_idx=2000):
        start = time.time()
        normalize_text_feature = -self.get_normalized_text_feature(query)
        end_feature = time.time()
        D, indices = self.k_nearest(normalize_text_feature, k=end_idx) # D is similarity scores
        end_search = time.time()
        print(f"{end_feature-start:.4f} for querying {query}. {end_search-end_feature} for computing KNN.")
        return D[start_idx:end_idx], indices[start_idx:end_idx], normalize_text_feature
    
    def grab_top_query_indices(self, query, start_idx=0, end_idx=2000):
        start = time.time()
        normalize_text_feature = self.get_normalized_text_feature(query)
        end_feature = time.time()
        D, indices = self.k_nearest(normalize_text_feature, k=end_idx) # D is similarity scores
        end_search = time.time()
        print(f"{end_feature-start:.4f} for querying {query}. {end_search-end_feature} for computing KNN.")
        return D[start_idx:end_idx], indices[start_idx:end_idx], normalize_text_feature
        
    def get_normalized_text_feature(self, query="a cat"):
        with torch.no_grad():
            text = clip.tokenize([query])
            text_feature = self.model.encode_text(text).numpy()
        return normalize(text_feature.astype(np.float32))

    def get_clip_score(self, image_path, query):
        image_feature = self.get_normalized_image_feature(image_path)
        text_feature = self.get_normalized_text_feature(query)
        return image_feature[0].dot(text_feature[0])
    
    def get_clip_score_feature(self, f_1, f_2):
        return f_1.dot(f_2)
    
    def get_normalized_image_feature(self, image_path):
        with torch.no_grad():
            image = self.preprocess(Image.open(image_path)).unsqueeze(0)
            image_feature = self.model.encode_image(image).numpy()
        return normalize(image_feature.astype(np.float32))

    def k_nearest(self, feature, k=4):
        chunk_iter = _chunk_iterator(k, 2048) # process every 2048 indices
        all_D, all_I = [], []
        
        mapping = list(range(self.total_feature_num))
        for chunk in chunk_iter:
            start = time.time()
            size_chunk = len(chunk)
            D, I = knn_ground_truth(
                       feature,
                       _path_iterator_for_numpy(self.clip_features_normalized_paths, mapping),
                       size_chunk
                   )
            end_search = time.time()
            all_D += list(D[0])
            print(f"{end_search-start:.4f} seconds for searching.")
            mapping, removed_indices = _remove_multiple_indices_at_once(mapping, list(I[0]))
            end_remove = time.time()
            print(f"{end_remove-start:.4f} seconds for one iteration.")
            all_I += removed_indices
        return all_D, all_I
    
    def k_nearest_meta(self, flickr_accessor, feature, k=4):
        """Return K Nearest scores + metadata list
        """
        # feature is length d numpy array of float32
        D, I = self.k_nearest(feature, k=k)
        meta_list = []
        for idx in I:
            meta_list.append(flickr_accessor[idx])
        return D, meta_list

        
class KNearestFaiss():
    def __init__(self, folder_path, model_name, use_chunked_memory=True, flickr_accessor=None):
        if type(flickr_accessor) == type(None):
            self.flickr_folders_path = _get_flickr_folder(folder_path)
            print(f"Loading from folder {self.flickr_folders_path}")
            self.flickr_folders = load_pickle(self.flickr_folders_path)
            self.flickr_accessor = FlickrAccessor(self.flickr_folders)
        else:
            self.flickr_accessor = flickr_accessor
        self.feature_name = get_feature_name(folder_path, model_name)
        self.normalize_features = load_pickle(self.feature_name)
        assert self.normalize_features.dtype == np.float32
        assert self.normalize_features.shape[0] == len(self.flickr_accessor)
        print(f"Size of dataset is {self.normalize_features.shape[0]}")
        self.use_chunked_memory = use_chunked_memory
        if self.normalize_features.shape[0] > MAX_SIZE and use_chunked_memory:
            print("Need to use brute force search without building an index")
        else:
            print(f"Cannot handle large size array > {MAX_SIZE}")

        if not self.use_chunked_memory:
            self.use_chunked_memory = False
            self.cpu_index = faiss.IndexFlatIP(self.normalize_features.shape[1])
            self.gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
                self.cpu_index
            )
            self.gpu_index.add(self.normalize_features)

        self.model_name = model_name
        self.model, self.preprocess = clip.load(self.model_name, device='cuda')
    
    def get_normalized_text_feature(self, query="a cat"):
        with torch.no_grad():
            text = clip.tokenize([query]).to('cuda')
            text_feature = self.model.encode_text(text).cpu().numpy()
        return normalize(text_feature.astype(np.float32))
    
    def get_text_difference_feature(self, query="a cat", diff_queries=['a dog', 'a car'], lmb=0.25):
        query_feature = self.get_normalized_text_feature(query=query)
        diff_query_features = [-lmb*self.get_normalized_text_feature(query=q)/len(diff_queries) for q in diff_queries]
        
        for diff_query_feature in diff_query_features:
            query_feature = query_feature + diff_query_feature
        return query_feature
    
    def get_clip_score(self, image_path, query, diff_queries=[], lmb=0.25):
        image_feature = self.get_normalized_image_feature(image_path)
        if len(diff_queries) > 0:
            text_feature = self.get_text_difference_feature(query=query, diff_queries=diff_queries, lmb=lmb)
        else:
            text_feature = self.get_normalized_text_feature(query)
        return image_feature[0].dot(text_feature[0])
    
    def get_clip_score_feature(self, f_1, f_2):
        return f_1.dot(f_2)
    
    def get_normalized_image_feature(self, image_path):
        with torch.no_grad():
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to('cuda')
            image_feature = self.model.encode_image(image).cpu().numpy()
        return normalize(image_feature.astype(np.float32))

    def k_nearest(self, feature, k=4):
        if self.use_chunked_memory:
            chunk_iter = _chunk_iterator(k, 2048)
            all_D, all_I = [], []
            mapping = list(range(self.normalize_features.shape[0]))
            for chunk in chunk_iter:
                # import pdb; pdb.set_trace()
                size_chunk = len(chunk)
                start = time.time()
                D, I = knn_ground_truth(feature, _matrix_iterator(self.normalize_features[mapping], MAX_SIZE), size_chunk)
                end_search = time.time()
                all_D += list(D[0])
                print(f"{end_search-start:.4f} seconds for searching.")
                # mapped_I = []
                # for idx in I[0]:
                #     mapped_I.append(mapping[idx])
                mapping, removed_indices = _remove_multiple_indices_at_once(mapping, list(I[0]))
                end_remove = time.time()
                print(f"{end_remove-start:.4f} seconds for one iteration.")
                all_I += removed_indices
        else:
            raise NotImplementedError()
            # D, I = self.gpu_index.search(feature, k) # actual search
            # all_D, all_I = D[0], I[0]
        return all_D, all_I

    
    def k_nearest_meta(self, feature, k=4):
        # feature is length d numpy array of float32
        D, I = self.k_nearest(feature, k=k)
        meta_list = []
        for idx in I:
            meta_list.append(self.flickr_accessor[idx])
        return D, meta_list

    def k_nearest_meta_and_clip_feature(self, feature, k=4):
        # feature is length d numpy array of float32
        D, I = self.k_nearest(feature, k=k)
        meta_list = []
        for idx in I:
            meta_list.append(self.flickr_accessor[idx])
        clip_features = self.normalize_features[I]
        return D, meta_list, clip_features
