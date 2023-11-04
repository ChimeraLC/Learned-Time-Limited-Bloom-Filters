import argparse
import json
import random
import torch
import os
import numpy as np
import pandas as pd

from runner import Runner
from sandwich_bloom import Learned_Bloom
from sandwich_time_bloom import Learned_AP_Bloom
from time_bloom import AP_Bloom

# Parse the arguments
def get_args():
    parser = argparse.ArgumentParser(description="Temporary Description")

    # General Args
    parser.add_argument("--seed", type=int, default=0)

    #  Data Args
    parser.add_argument("--valid_data_path", type=str, default="data/player-stats-abriv.csv") #data from https://github.com/lukearend/osrs-hiscores
    parser.add_argument("--invalid_data_path", type=str, default="data/false-names.csv")
    parser.add_argument("--model_save_path", type=str, default="saved-model.pth")
    parser.add_argument("--retrain_model", type=bool, default = False)
    parser.add_argument("--max_len", type=int, default = 12)
    parser.add_argument("--chars", type=str, default="qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890_[]#- ")
    parser.add_argument("--train_ratio", type=float, default=0.7)

    # Model Args: Load JSON file
    parser.add_argument('--model_args', type=str, default="args.json")

    args = parser.parse_args()

    with open(args.model_args, 'rt') as f:
        args.__dict__.update(json.load(f))

    return args

# Fix the seed
def seed_all(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)

def run(args):
    runner = Runner(args)
    runner.train()
    return runner

if __name__ == "__main__":
    args = get_args()

    # Seed
    seed_all(args.seed)

    # Train/Load model based on username dataset
    runner = run(args)

    # Create default and learned age partitioned bloom filters
    ap_filter = AP_Bloom(3, 2, 100)
    l_ap_filter = Learned_AP_Bloom(3, 2, 100, runner)

    # Get 2000 usernames not used during training
    data = pd.read_csv("data\player-stats.csv")
    incl_names = data['username'][300000:302000].values
    # Insert first 300 usernames into filter
    for name in incl_names[:300]:
        ap_filter.insert(name)
        l_ap_filter.insert(name)

    # Check inclusion rate
    included = 0
    l_included = 0
    for name in incl_names[:300]:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("Out of 300 elements,", included, "were correctly included in default filter")
    print("Out of 300 elements,", l_included, "were correctly included in learned filter")

    # Insert 200 more usernames
    for name in incl_names[300:500]:
        ap_filter.insert(name)
        l_ap_filter.insert(name)

    # Check inclusion rate
    included = 0
    l_included = 0
    for name in incl_names[:200]:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("Out of the first 200 elements,", included, "were still included in default filter")
    print("Out of the first 200 elements,", l_included, "were still included in learned filter")

    # Check inclusion rate
    included = 0
    l_included = 0
    for name in incl_names[200:500]:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("Out of 300 recent elements,", included, "were correctly included in default filter")
    print("Out of 300 recent elements,", l_included, "were correctly included in learned filter")


    # Get 10000 other usernames
    excl_names = data['username'][302000:312000].values
    # Check inclusion rate
    included = 0
    l_included = 0
    for name in excl_names:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("There is a false positive rate of", included/10000, "among common usernames in default filter")
    print("There is a false positive rate of", l_included/10000, "among common usernames in learned filter")

    
    # Get 10000 other random strings
    alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 "
    included = 0
    l_included = 0
    for _ in range(10000):
        length = random.randint(5, 12)
        username = "".join(random.choices(alphabet, k = length))
        if (ap_filter.query(username)):
            included += 1
        if (l_ap_filter.query(username)):
            l_included += 1
    print("There is a false positive rate of", included/10000, "among random usernames in default filter")
    print("There is a false positive rate of", l_included/10000, "among random usernames in learned filter")

