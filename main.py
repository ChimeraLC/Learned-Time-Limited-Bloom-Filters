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

def test(runner, real_percent, size):
    print("Running test with real percentage of:", real_percent)

    ap_filter = AP_Bloom(3, 2, size)
    l_ap_filter = Learned_AP_Bloom(3, 2, size, runner)
    # Calculate size of each type
    real_size = int(real_percent * size * 10)
    unreal_size = 10 * size - real_size
    real_data = pd.read_csv("data\player-stats.csv") # Realistic names
    unreal_data = pd.read_csv("data/false-names.csv") # Unrealistic names
    incl_names = np.concatenate((real_data['username'][20000:20000 + real_size].values,
                            unreal_data['username'][20000:20000 + real_size].values))
    # Shuffle names
    np.random.shuffle(incl_names)
    # Insert first few usernames into filter
    for name in incl_names[:3 * size]:
        ap_filter.insert(name)
        l_ap_filter.insert(name)

    # Check inclusion rate
    included = 0
    l_included = 0
    for name in incl_names[:3 * size]:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("Out of", 3 * size, "elements,", included, "were correctly included in default filter")
    print("Out of", 3 * size, "elements,", l_included, "were correctly included in learned filter")

    # Insert more usernames
    for name in incl_names[3 * size:5 * size]:
        ap_filter.insert(name)
        l_ap_filter.insert(name)

    # Check inclusion rate
    included = 0
    l_included = 0
    for name in incl_names[:2 * size]:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("Out of the first", 2 * size, "elements,", included, "were still included in default filter")
    print("Out of the first", 2 * size, "elements,", l_included, "were still included in learned filter")

    # Check inclusion rate
    included = 0
    l_included = 0
    for name in incl_names[2 * size: 5 * size]:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("Out of", 3 * size, "recent elements,", included, "were correctly included in default filter")
    print("Out of", 3 * size, "recent elements,", l_included, "were correctly included in learned filter")


    # Get other usernames
    excl_names = incl_names[5 * size:]
    # Check inclusion rate
    included = 0
    l_included = 0
    for name in excl_names:
        if (ap_filter.query(name)):
            included += 1
        if (l_ap_filter.query(name)):
            l_included += 1
    print("There is a false positive rate of", included/len(excl_names), "among common usernames in default filter")
    print("There is a false positive rate of", l_included/len(excl_names), "among common usernames in learned filter")

    print("The default filter has size", ap_filter.get_size())
    print("The learned filter has size", l_ap_filter.get_size())

if __name__ == "__main__":
    args = get_args()

    # Seed
    seed_all(args.seed)

    # Train/Load model based on username dataset
    runner = run(args)

    # Run a few different percentages
    for percentage in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        test(runner, percentage, 100)