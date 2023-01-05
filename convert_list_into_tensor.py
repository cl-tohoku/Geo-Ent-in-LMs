import argparse
import io,sys
import os
import csv
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from tqdm import tqdm
from itertools import chain


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path_list", required=True, nargs='*', 
                            help="")
    args = parser.parse_args()
    return args


def convert_list_into_tensor(path_list):
    print("multiple_load_tensor")
    emb_list = []
    for path in path_list:
        print(f'loading : {path}')
        emb_list = torch.load(path)
        emb_tensor = torch.stack(emb_list, dim=0)
        path_without_ext = str(path).replace('_list.pt', '')
        torch.save(emb_tensor, path_without_ext +  "_tensor.pt")

args = get_args()
convert_list_into_tensor(args.emb_path_list)
