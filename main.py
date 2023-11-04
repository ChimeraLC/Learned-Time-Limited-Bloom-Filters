import argparse
import json
import random
import torch
import os
import numpy as np
import pandas as pd

from runner import Runner
from sandwich_bloom import Learned_Bloom

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

    # Create learned bloom filter (currently using unoptimized parameters)
    b_filter = Learned_Bloom(8196, 4, 512, 2, runner)

    # Get 2000 usernames not used during training
    data = pd.read_csv("data\player-stats.csv")
    incl_names = data['username'][300000:302000].values
    # Insert usernames into filter
    for name in incl_names:
        b_filter.insert(name)

    # Check inclusion rate
    included = 0
    for name in incl_names:
        if (b_filter.test(name)):
            included += 1
    print("Out of 2000 elements,", included, "were correctly included")

    # Get 10000 other usernames
    excl_names = data['username'][302000:312000].values
    # Check inclusion rate
    included = 0
    for name in excl_names:
        if (b_filter.test(name)):
            included += 1
    print("There is a false positive rate of", included/10000, "among common usernames")

    # Get 10000 other random strings
    alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 "
    included = 0
    for _ in range(10000):
        length = random.randint(5, 12)
        username = "".join(random.choices(alphabet, k = length))
        if (b_filter.test(username)):
            included += 1
    print("There is a false positive rate of", included/10000, "among random usernames")
