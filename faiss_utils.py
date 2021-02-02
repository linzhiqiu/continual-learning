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

def get_flickr_folder(folder_path):
    return os.path.join(folder_path, "all_folders.pickle")

def get_feature_name(folder_path, model_name, normalize=True):
    if normalize:
        normalize_str = "_normalized"
    else:
        normalize_str = ""
    return os.path.join(folder_path, f"features_{model_name.replace(os.sep, '_')}{normalize_str}.pickle")

def matrix_iterator(matrix, chunk_size):
    for i0 in range(0, matrix.shape[0], chunk_size):
        yield matrix[i0:i0 + chunk_size]

def chunk_iterator(length, chunk_size):
    lst = list(range(length))
    # if chunk_size < length:
    #     yield lst
    # else:
    for i0 in range(0, length, chunk_size):
        yield lst[i0:i0 + chunk_size]

def remove_multiple_indices_at_once(lst, removed_indices):
    mapped_removed_indices = [lst[j] for j in removed_indices]
    
    for ele in sorted(removed_indices, reverse=True):  
        del lst[ele]
    # new_lst = [i for j, i in enumerate(lst) if j not in removed_indices]
    return lst, mapped_removed_indices

class ResultHeap:
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
    rh = ResultHeap(nq, k)

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

class KNearestFaiss():
    def __init__(self, folder_path, model_name, use_chunked_memory=True):
        self.feature_name = get_feature_name(folder_path, model_name)
        self.normalize_features = load_pickle(self.feature_name)
        assert self.normalize_features.dtype == np.float32
        self.flickr_folders_path = get_flickr_folder(folder_path)
        self.flickr_folders = load_pickle(self.flickr_folders_path)
        self.flickr_accessor = FlickrAccessor(self.flickr_folders)
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
    
    def get_text_difference_feature(self, query="a cat", diff_queries=['a dog', 'a car'], lmb=0.5):
        query_feature = self.get_normalized_text_feature(query=query)
        diff_query_features = [-lmb*self.get_normalized_text_feature(query=q)/len(diff_queries) for q in diff_queries]
        
        for diff_query_feature in diff_query_features:
            query_feature = query_feature - diff_query_feature
        return query_feature
    
    def get_clip_score(self, image_path, query, diff_queries=[], lmb=0.5):
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
            chunk_iter = chunk_iterator(k, 2048)
            all_D, all_I = [], []
            mapping = list(range(self.normalize_features.shape[0]))
            for chunk in chunk_iter:
                # import pdb; pdb.set_trace()
                size_chunk = len(chunk)
                start = time.time()
                D, I = knn_ground_truth(feature, matrix_iterator(self.normalize_features[mapping], MAX_SIZE), size_chunk)
                end_search = time.time()
                all_D += list(D[0])
                print(f"{end_search-start:.4f} seconds for searching.")
                # mapped_I = []
                # for idx in I[0]:
                #     mapped_I.append(mapping[idx])
                mapping, removed_indices = remove_multiple_indices_at_once(mapping, list(I[0]))
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