### ----------------------------------------------
def parse_dwca(dwca_avh_path):
   from dwca.read import DwCAReader
   from dwca.darwincore.utils import qualname as qn
   # Let's open our archive...
   # Using the with statement ensure that resources will be properly freed/cleaned after use.
   with DwCAReader(dwca_avh_path) as dwca:
      dwca.metadata
      print("Core type is: %s" % dwca.descriptor.core.type)
      core_df = dwca.pd_read('occurrence.txt', parse_dates=True)
   return core_df

def subset_records_genus(core_df, target_genus):
   # nominate a subset of samples, by genus
   target_df = core_df.loc[core_df['genus'] == target_genus]
   return target_df






def get_target_images(target_df, img_path, img_ext, dest_dir):
   import urllib.request
   import requests
   cat_nums = target_df['catalogNumber']
   cat_nums_proc= cat_nums.str.split().str[0] + cat_nums.str.split().str[1]
   #for i in range(len(cat_nums_proc)):
   for i in range(20):
      cat_num_i = cat_nums_proc.iloc[i]
      img_url_i = img_path + cat_num_i + img_ext
      img_dest_i = dest_dir + cat_num_i + img_ext
      print(img_url_i)
      #urllib.request.urlretrieve(img_url_i, img_dest_i)
      request = requests.get(img_url_i)
      if request.status_code == 200:
         print( 'catalog number: ' + cat_num_i + ' image file downloading')
         urllib.request.urlretrieve(img_url_i, img_dest_i)
      else:
         print('Web site does not exist')


def get_images(target_df, img_path, img_ext, dest_dir, get_file=True, use_max=False, max_num=50):
   import urllib.request
   import requests
   cat_nums = target_df['catalogNumber']
   cat_nums_proc= cat_nums.str.split().str[0] + cat_nums.str.split().str[1]
   genout = []
   sppout = []
   nswout = []
   allout = []
   if bool(use_max):
     mn = max_num
   else:   
     mn = len(cat_nums_proc)
   for i in range(mn):
      cat_num_i = cat_nums_proc.iloc[i]
      img_url_i = img_path + cat_num_i + img_ext
      img_dest_i = dest_dir + cat_num_i + img_ext
      print(img_url_i)
      #urllib.request.urlretrieve(img_url_i, img_dest_i)
      request = requests.head(img_url_i)
      if request.status_code == 200:
         print( 'catalog number: ' + cat_num_i + ' exists')
         nswout.append(cat_num_i)
         genout.append(target_df.iloc[i,68])
         sppout.append(target_df.iloc[i,69])
         allout.append(target_df.iloc[i,])
         if bool(get_file): 
            urllib.request.urlretrieve(img_url_i, img_dest_i)
            print( 'catalog number: ' + cat_num_i + ' image downloading')
      else:
            print( 'catalog number: ' + cat_num_i + ' does not exist')
   return nswout, genout, sppout, allout 

### ----------------------------------------------


