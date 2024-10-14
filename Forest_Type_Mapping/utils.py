import pandas as pd
import argparse


def print_info(df: pd.DataFrame):
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.columns)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
