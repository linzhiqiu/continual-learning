import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil import parser


def plot_scores_jupyter(score_list, plot_mean=False):
    min_count = None
    max_count = None
    avg_count = 0
    for s in score_list:
        if not min_count or count < min_count:
            min_count = s
        if not max_count or count > max_count:
            max_count = s
        avg_count += s
    avg_count = avg_count / len(score_list)

    bins = np.linspace(min_count, max_count, 15)
    plt.figure(figsize=(8,8))
    if plot_mean:
        plt.axhline(y=avg_count, label=f"Mean scores of Samples {avg_count}", linestyle='--', color='black')
    plt.hist(y, bins, alpha=0.5, label='Number of samples')
    plt.legend(loc='upper right')
    plt.tight_layout()


def plot_time_jupyter(metadata_list, mode='year', date='date_uploaded', plot_mean=False):
    buckets_dict = {}

    for metadata in metadata_list:
        meta = metadata.get_metadata()
        if date == 'date_uploaded':
            date_obj = datetime.utcfromtimestamp(int(meta.DATE_UPLOADED))
        else:
            raise NotImplementedError()

        if mode == 'year':
            time_meta = date_obj.year
        elif mode == 'month':
            time_meta = (date_obj.year, date_obj.month)
        
        if time_meta in buckets_dict:
            buckets_dict[time_meta].append(meta)
        else:
            buckets_dict[time_meta] = [meta]
    
    if mode == 'year' and not 2004 in buckets_dict:
        buckets_dict[2004] = []
        
    sorted_buckets_list = sorted(buckets_dict.keys())
    min_count = None
    max_count = None
    avg_count = 0
    for b in sorted_buckets_list:
        count = len(buckets_dict[b])
        if not min_count or count < min_count:
            min_count = count
        if not max_count or count > max_count:
            max_count = count
        avg_count += count
    avg_count = avg_count / len(sorted_buckets_list)
    
    # print(f"Min/Max number of images {optional_name} per bucket: {min_count}/{max_count}. Average is {avg_count}")
    plt.figure(figsize=(8,8))
    axes = plt.gca()
    axes.set_ylim([0,max_count])
    plt.title(f'Number of samples per {mode}.')
    x = [str(a) for a in sorted_buckets_list]
    y = [len(buckets_dict[b]) for b in sorted_buckets_list]
    plt.bar(x, y, align='center')
    if plot_mean:
        plt.axhline(y=avg_count, label=f"Mean Number of Samples {avg_count}", linestyle='--', color='black')
    x_tick = [str(a) for a in sorted_buckets_list]
    plt.xticks(x, x_tick)
    plt.xlabel('Date')
    plt.ylabel(f'Number of samples for each {mode}')
    plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='large')
    plt.legend()
    plt.show()
    plt.close('all')
