import os, csv
import pandas as pd
import argparse
import itertools

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str)
parser.add_argument("--fields", help="Fields to join from metadata", action="append", type=str)
args = parser.parse_args()

raw_out = pd.read_csv(os.path.join(args.base_dir, "final_results.csv"), on_bad_lines='warn')
raw_dupe = pd.read_csv(os.path.join(args.base_dir, "test_duplicates_out.csv"))
joined_out = os.path.join(args.base_dir, "joined_final_results.csv")

meta_euc = pd.read_csv("/home/botml/euc/data/meta/euc.csv")
meta_cor = pd.read_csv("/home/botml/euc/data/meta/cor.csv")
meta_ang = pd.read_csv("/home/botml/euc/data/meta/ang.csv")
meta_euccor = pd.concat([meta_euc, meta_cor])
meta_joined = pd.concat([meta_euccor, meta_ang])

raw_dupe['id'] = raw_dupe['id'].str.strip()
raw_dupe['id'] = raw_dupe['id'].str.replace('NSW','NSW:NSW:NSW ')
raw_dupe['id'] = raw_dupe['id'].str.replace('.jpg','', regex=True)

raw_out['id']=raw_out['id'].str.strip()
metadf_subset = meta_joined[meta_joined.columns.intersection(args.fields)]

joined_results_1 = pd.merge(raw_out, metadf_subset, on='id', how='left')
joined_results_2 = pd.merge(joined_results_1, raw_dupe, on=['id', 'index'], how='left')
joined_results_rmna = joined_results_2.dropna()

joined_results_rmna.to_csv(joined_out, sep=',', index=False)


