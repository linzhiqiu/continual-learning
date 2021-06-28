
import os
import argparse
import shutil

import math
import time

import csv
import pickle
import random

import numpy as np
import json

MTURK_QUERY_DICT = {'computer': 'laptop',
                'camera': 'camera',
                'bus' : 'bus',
                'sweater' : 'sweater',
                'shirt' : 'polo shirt',
                'racing' : 'racing',
                'hockey' : 'hockey',
                'cosplay' : 'cosplay',
                'baseball' : 'baseball',
                'tennis' : 'tennis',}

argparser = argparse.ArgumentParser()
argparser.add_argument('--folder',
                    #    default='~/Documents/yfcc_mturk',
                       default='./',
                       help='main folder')
argparser.add_argument("--original_csv",
                       default='original_200.csv',
                       help="The original csv")
argparser.add_argument("--mturk_csv_ethics",
                       default='Batch_4428513_batch_results_ethics.csv',
                       help="New csv with ethics result")
argparser.add_argument("--mturk_csv_validation",
                       default='Batch_4428514_batch_results_validation.csv',
                       help="New csv with validation result")

def parse_yes_or_no(s):
    if s == 'Yes':
        return True
    elif s == 'No':
        return False
    else:
        raise NotImplementedError()

class Result():
    def __init__(self, row, input_prefix="Input."):
        self.ID = row[f'{input_prefix}ID']
        self.bucket_index = row[f'{input_prefix}bucket_index']
        # self.input_query = MTURK_QUERY_DICT[row[f'{input_prefix}query']]
        self.input_query = row[f'{input_prefix}query']
        self.image_url = row[f'{input_prefix}image_url']

class EthicsResult(Result):
    def __init__(self, row):
        super().__init__(row)
        self.answer = parse_yes_or_no(row['Answer.image-contains.label'])

class ValidationResult(Result):
    def __init__(self, row):
        super().__init__(row)
        self.answer_list = parse_label_list(row['Answer.taskAnswers'])


def parse_label_list(s):
    lst_of_labels = json.loads(s)[0]['image']['labels']
    lst_of_labels = [MTURK_QUERY_DICT[label] for label in lst_of_labels]
    return lst_of_labels

def has_same_input(result_a, result_b):
    return result_a.ID == result_b.ID \
            and result_a.bucket_index == result_b.bucket_index \
            and result_a.image_url ==result_b.image_url \
            and result_a.input_query == result_b.input_query

def parse_single_result(result_list):
    result_dict =  {
        'image_url': result_list[0].image_url,
        'ID': result_list[0].ID,
        'bucket_index': result_list[0].bucket_index,
        'query': result_list[0].input_query,
    }
    for idx, result in enumerate(result_list):
        result_dict[f"worker_{idx}"] = json.dumps(result.answer_list)
    return result_dict

def parse_validation_result(mturk_results_validation, save_csv_dir=None):
    all_negatives_with_majority_positive = []
    all_positive_with_majority_wrong = [] # majority is not agreeing on the single correct class
    all_positives_with_majority_negative = [] # majority select NEG
    all_positives_with_majority_wrong_class = [] # majority select some other classes
    
    all_positives_with_majority_multi_select = []  # majority select correct + some other classes
    true_negative_counts = 0
    true_positive_counts = 0
    worker_num = None
    for ID in mturk_results_validation:
        if not worker_num:
            worker_num = len(mturk_results_validation[ID])
        # yes_count = 0
        result_0 = mturk_results_validation[ID][0]
        if result_0.input_query == "NEGATIVE":
            true_negative_counts += 1
            positive_votes = 0
            for result in mturk_results_validation[ID]:
                if len(result.answer_list) > 0:
                    positive_votes += 1
            if positive_votes > int(worker_num/2):
                result_dict = parse_single_result(mturk_results_validation[ID])
                result_dict['disagree_count'] = positive_votes
                all_negatives_with_majority_positive.append(result_dict)
        else:
            true_positive_counts += 1
            votes = {
                'negative' : 0, # choose no class
                'correct' : 0, # choose single correct class
                'correct_but_with_other_class' : 0, # Choose other classes while choosing correct one
                'wrong_class' : 0,
            }
            for result in mturk_results_validation[ID]:
                if len(result.answer_list) == 0:
                    votes['negative'] += 1
                elif len(result.answer_list) == 1:
                    if result.answer_list[0] == result.input_query:
                        votes['correct'] += 1
                    else:
                        votes['wrong_class'] += 1
                else:
                    has_correct = False
                    for answer in result.answer_list:
                        if answer == result.input_query:
                            votes['correct_but_with_other_class'] += 1
                            has_correct = True
                            break
                    if not has_correct:
                        votes['wrong_class'] += 1

            assert sum([votes[k] for k in votes]) == worker_num
            sorted_keys = sorted(list(votes.keys()), key=lambda k: votes[k], reverse=True)
            result_dict = parse_single_result(mturk_results_validation[ID])

            if sorted_keys[0] != 'correct':
                result_dict['disagree_count'] = votes[sorted_keys[0]]
                all_positive_with_majority_wrong.append(result_dict)
            else:
                result_dict['disagree_count'] = worker_num - votes['correct']
            if votes['negative'] > int(worker_num/2):
                all_positives_with_majority_negative.append(result_dict)
            elif votes['correct_but_with_other_class'] > int(worker_num/2):
                all_positives_with_majority_multi_select.append(result_dict)
            elif votes['wrong_class'] > int(worker_num/2):
                all_positives_with_majority_wrong_class.append(result_dict)

            
    print(f"True Positive: {true_positive_counts}")
    print(f"True Negative: {true_negative_counts}")
    print(f"True Negative with positive majority: {len(all_negatives_with_majority_positive)}")
    print(f"True Negative with majority not selecting the single correct class: {len(all_positive_with_majority_wrong)}")
    print(f"True Positive with majority select no answer: {len(all_positives_with_majority_negative)}")
    print(f"True Positive with majority select wrong class(es): {len(all_positives_with_majority_wrong_class)}")
    print(f"True Positive with majority select multiple answers + correct: {len(all_positives_with_majority_multi_select)}")
    all_results = all_negatives_with_majority_positive \
                + all_positive_with_majority_wrong
    all_results = sorted(all_results, key=lambda l:l['disagree_count'], reverse=True)
    all_positive_with_majority_wrong = sorted(all_positive_with_majority_wrong, key=lambda l:l['disagree_count'], reverse=True)
    all_negatives_with_majority_positive = sorted(all_negatives_with_majority_positive, key=lambda l:l['disagree_count'], reverse=True)
    all_positives_with_majority_negative = sorted(all_positives_with_majority_negative, key=lambda l:l['disagree_count'], reverse=True)
    all_positives_with_majority_other_class = sorted(all_positives_with_majority_wrong_class, key=lambda l:l['disagree_count'], reverse=True)
    all_positives_with_majority_multi_select = sorted(all_positives_with_majority_multi_select, key=lambda l:l['disagree_count'], reverse=True)
    headers = ['image_url', 'ID', 'bucket_index', 'query', 'disagree_count'] + [f"worker_{idx}" for idx in range(worker_num)]
    if save_csv_dir:
        save_csv_path = os.path.join(save_csv_dir, "all_results.csv")
        save_csv(headers, all_results, save_csv_path)
        save_csv_path_true_neg_wrong = os.path.join(save_csv_dir, "true_negative_wrongs.csv")
        save_csv(headers, all_negatives_with_majority_positive, save_csv_path_true_neg_wrong)
        save_csv_path_true_pos_wrong = os.path.join(save_csv_dir, "true_pos_wrongs.csv")
        save_csv(headers, all_positive_with_majority_wrong, save_csv_path_true_pos_wrong)
        save_csv_path_true_pos_wrong_by_neg = os.path.join(save_csv_dir, "true_pos_wrongs_by_neg.csv")
        save_csv(headers, all_positives_with_majority_negative, save_csv_path_true_pos_wrong_by_neg)
        save_csv_path_true_pos_wrong_by_multi = os.path.join(save_csv_dir, "true_pos_wrongs_by_multi.csv")
        save_csv(headers, all_positives_with_majority_multi_select, save_csv_path_true_pos_wrong_by_multi)
        save_csv_path_true_pos_wrong_by_wrong = os.path.join(save_csv_dir, "true_pos_wrongs_by_other.csv")
        save_csv(headers, all_positives_with_majority_wrong_class, save_csv_path_true_pos_wrong_by_wrong)
    return all_results

def parse_ethics_result(mturk_results_ethics, save_csv_path=None):
    all_unethics = []
    for ID in mturk_results_ethics:
        yes_count = 0
        for result in mturk_results_ethics[ID]:
            if result.answer:
                yes_count += 1
        if yes_count > 0:
            all_unethics.append({
                'image_url': result.image_url,
                'ID': result.ID,
                'bucket_index': result.bucket_index,
                'query': result.input_query,
                'yes_count' : yes_count
            })
    all_unethics = sorted(all_unethics, key=lambda l:l['yes_count'], reverse=True)
    headers = ['image_url', 'ID', 'bucket_index', 'query', 'yes_count']
    if save_csv_path:
        save_csv(headers, all_unethics, save_csv_path)
    return all_unethics

def save_csv(headers, results, save_csv_path):
    with open(save_csv_path, 'w', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for csv_dict in results:
            writer.writerow(csv_dict)
    print(f"Write at {save_csv_path}")

if __name__ == "__main__":
    args = argparser.parse_args()
    start = time.time()

    original_csv = os.path.join(args.folder, args.original_csv)
    mturk_csv_ethics = os.path.join(args.folder, args.mturk_csv_ethics)
    mturk_csv_validation = os.path.join(args.folder, args.mturk_csv_validation)
    
    mturk_results_ethics = {} #key is ID, value is a list of result
    with open(mturk_csv_ethics, newline='\n') as mturk_file:
        mturk_reader = csv.DictReader(mturk_file)
        for row in mturk_reader:
            if row['Input.ID'] not in mturk_results_ethics:
                mturk_results_ethics[row['Input.ID']] = [EthicsResult(row)]
            else:
                mturk_results_ethics[row['Input.ID']].append(EthicsResult(row))
        
        for ID in mturk_results_ethics:
            assert len(mturk_results_ethics[ID]) == len(mturk_results_ethics[row['Input.ID']])
    
    parsed_ethics_csv_path = os.path.join(args.folder, args.mturk_csv_ethics[:-4]+"_result.csv")
    all_unethics = parse_ethics_result(mturk_results_ethics, parsed_ethics_csv_path)

    mturk_results_validation = {} #key is ID, value is a list of result
    with open(mturk_csv_validation, newline='\n') as mturk_file:
        mturk_reader = csv.DictReader(mturk_file)
        for row in mturk_reader:
            if row['Input.ID'] not in mturk_results_validation:
                mturk_results_validation[row['Input.ID']] = [ValidationResult(row)]
            else:
                mturk_results_validation[row['Input.ID']].append(ValidationResult(row))
        
        for ID in mturk_results_validation:
            assert len(mturk_results_validation[ID]) == len(mturk_results_validation[row['Input.ID']])

    parsed_validation_csv_dir = os.path.join(args.folder, args.mturk_csv_validation[:-4])
    if not os.path.exists(parsed_validation_csv_dir): os.makedirs(parsed_validation_csv_dir)
    all_invalids = parse_validation_result(mturk_results_validation, parsed_validation_csv_dir)


    with open(original_csv, newline='\n') as original_file:
        original_reader = csv.DictReader(original_file)
        for row in original_reader:
            assert row["ID"] in mturk_results_ethics
            assert row["ID"] in mturk_results_validation
            result = Result(row, input_prefix="")
            if not has_same_input(result, mturk_results_validation[row['ID']][0]):
                import pdb; pdb.set_trace()
                has_same_input(result, mturk_results_validation[row['ID']][0])
            if not has_same_input(result, mturk_results_ethics[row['ID']][0]):
                import pdb; pdb.set_trace()
        
    

