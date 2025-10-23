import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import check_path, clean_text, amazon18_dataset2fullname,write_json_file,write_remap_index

def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters


def load_meta_items(file):
    items = {}
    # re_tag = re.compile('</?\w+[^>]*>')
    with gzip.open(file, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            data = json.loads(line)
            item = data["asin"]
            title = clean_text(data["title"])

            descriptions = data["description"]
            descriptions = clean_text(descriptions)
            # new_descriptions = []
            # for description in descriptions:
            #     description = re.sub(re_tag, '', description)
            #     new_descriptions.append(description.strip())
            # descriptions = " ".join(new_descriptions).strip()

            brand = data["brand"].replace("by\n", "").strip()

            categories = data["category"]
            new_categories = []
            for category in categories:
                if "</span>" in category:
                    break
                new_categories.append(category.strip())
            categories = ",".join(new_categories[1:]).strip()

            items[item] = {"title": title, "description": descriptions, "brand": brand, "categories": categories}
            # print(items[item])
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates( # users is set
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters:
            if inter[1] in interacted_item: # 过滤重复交互
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon18_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))

    rating_inters = make_inters_in_order(rating_inters)

    rating_inters = filter_inters(rating_inters, can_items=meta_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters, meta_items

def save_inter(args, inters):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.inter'), 'w') as file:
        file.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        for inter in inters:
            user, item, rating, timestamp = inter
            file.write(f'{user}\t{item}\t{rating}\t{timestamp}\n')


def save_feat(args, feat, all_items):
    iid_list = list(feat.keys())
    num_item = 0
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.item'), 'w') as file:
        # "title": title, "description": descriptions, "brand": brand, "categories": categories
        file.write('item_id:token\ttitle:token_seq\tbrand:token\tcategories:token_seq\n')
        for iid in iid_list:
            if iid in all_items:
                num_item += 1
                title, brand, categories = feat[iid]["title"], feat[iid]["brand"], feat[iid]["categories"]
                file.write(f'{iid}\t{title}\t{brand}\t{categories}\n')
    print("num_item: ", num_item)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters, meta_items = preprocess_rating(args)

    check_path(os.path.join(args.output_path, args.dataset))


    all_items = set()
    for inter in rating_inters:
        user, item, rating, timestamp = inter
        all_items.add(item)

    print("total item: ", len(list(all_items)))

    save_inter(args,rating_inters)
    save_feat(args,meta_items, all_items)


