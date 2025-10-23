import collections
import json
import logging
import os
import torch
import numpy as np
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from datasets import EmbDataset
from models.clvae import CLVAE

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

def parse_args():
    parser = argparse.ArgumentParser(description="Run CLVAE model with collision handling")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="The checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory")
    parser.add_argument("--output_file", type=str, default="output.index.json", help="The output JSON file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    return parser.parse_args()

def main():
    args = parse_args()

    output_file = os.path.join(args.output_dir, args.output_file)

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    print("Processing data")
    # data = EmbDataset(model_args.data_path)
    data = EmbDataset(args.dataset)
    print("data length:", len(data))

    CODE_SIZE = 1024

    model = CLVAE(in_dim=data.dim,
                  num_emb_list=[CODE_SIZE,CODE_SIZE,CODE_SIZE,CODE_SIZE],
                  e_dim=model_args.e_dim,
                  layers=model_args.layers,
                  dropout_prob=model_args.dropout_prob,
                  bn=model_args.bn,
                  loss_type=model_args.loss_type,
                  quant_loss_weight=model_args.quant_loss_weight,
                  kmeans_init=model_args.kmeans_init,
                  kmeans_iters=model_args.kmeans_iters,
                  sk_epsilons=model_args.sk_epsilons,
                  sk_iters=model_args.sk_iters)

    print("Loading model checkpoint...")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    all_indices = []
    all_indices_str = []
    # prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(ind+1+i*CODE_SIZE)

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(len(collision_item_groups))
        # import pdb;pdb.set_trace()
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    # code.append(prefix[i].format(int(ind)))
                    code.append(ind+1+i*CODE_SIZE)

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp)

if __name__ == "__main__":
    main()
