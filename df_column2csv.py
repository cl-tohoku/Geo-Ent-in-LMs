import pandas as pd
import numpy as np
import argparse
import os
import collections

pd.set_option('display.max_rows', 100)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=os.path.abspath, 
                        help="input dataset path")
    parser.add_argument("--column_name", type=str, 
                        help="column name")
    parser.add_argument("--output", type=os.path.abspath, 
                       help="input dataset path")
    args = parser.parse_args()
    return args

def read_json(path):
    print('json reading')
    return pd.read_json(path, orient='records', lines=True)


args = get_args()
print(f'input path: {args.input}')
df = read_json(args.input)

print(f'savefile path: {args.output}')
if column_name is not None :
    output_df = pd.DataFrame([df[args.column_name]]).T
    output_df.to_csv(args.output, encoding="utf_8_sig")
else :
    df.to_csv(args.output, encoding="utf_8_sig")