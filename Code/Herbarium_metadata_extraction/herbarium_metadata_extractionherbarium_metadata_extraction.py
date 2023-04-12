import sys
sys.path.append("/home/botml/getImages/utils")
from nsw_img_utils import *
img_path = "https://herbariumnsw-pds.s3-ap-southeast-2.amazonaws.com/images/"

# use .jpg or .jp2
img_ext = ".jpg"

# local copy of dwca nsw data -- see ReadMe.txt above
dwca_avh_path = '/home/botml/getImages/nsw/dwca-nsw_avh-v1.0.zip'

# if retrieving images for a specific genus

img_dir =  "/home/botml/euc/data/raw/"
meta_dir = "/home/botml/euc/data/meta/"

cdf = parse_dwca(dwca_avh_path)

euc_df = subset_records_genus(cdf, 'Eucalyptus')
cor_df = subset_records_genus(cdf, 'Corymbia')
ang_df = subset_records_genus(cdf, 'Angophora')

import pandas as pd
euc_df.to_csv(r'/home/botml/euc/data/meta/euc.csv', index = False)
cor_df.to_csv(r'/home/botml/euc/data/meta/cor.csv', index = False)
ang_df.to_csv(r'/home/botml/euc/data/meta/ang.csv', index = False)

nsw, gen, spp, all = get_images(ang_df, img_path, img_ext, img_dir, get_file=True, use_max=False)

nsw, gen, spp, all = get_images(cor_df, img_path, img_ext, img_dir, get_file=True, use_max=False)

nsw, gen, spp, all = get_images(euc_df, img_path, img_ext, img_dir, get_file=True, use_max=False)

