import os, glob, sys
import math, csv
sys.path.append("/home/botml/code/dev")
import linking_functions
import pandas as pd
import shutil
import contextlib

base_dir = os.getcwd()
print ("Base directory set as: ", base_dir)

#Making folders
temp_pred = os.path.join(base_dir, "temp_pred/")
temp_pred_leaf = os.path.join(base_dir, "temp_pred_leaf/subf/")
temp_image_subset = os.path.join(base_dir, "temp_image_subset")

os.makedirs(temp_pred)
os.makedirs(temp_pred_leaf)
print ("Making directories: /temp_pred, /temp_pred_leaf")
os.makedirs(temp_image_subset)
print ("Making directories: /temp_image_subset")

#Making files
temp_filter = open('temp_filter.csv', 'x')
temp_filter.close
temp_classifier_results_test = open('temp_classifier_results_test.csv', 'x')
temp_classifier_results_test.close
print ("Making files: /classifier_results_test.csv, /temp_filter.csv")

#File list of all images - making batches
file_list = glob.glob('/home/botml/euc/data/raw/*.jpg') #replace dir with image dir, could possibly do this as a argpars
#file_list = glob.glob('/home/karina/test/classifier_training_data/*.jpg')
temp_image_subset_dir = os.path.join(base_dir, "temp_image_subset")  
batch_size = math.ceil(len(file_list)/5)

image_length = len(file_list)

counter = 1
for img in file_list:
	if ((counter*batch_size)+1) < image_length:
		batch_start = batch_size * (counter-1) + 1
		batch_end = batch_size * counter
		batched = file_list[batch_start:batch_end]
		out_file = f"file_list_batch_{counter}.txt"
		out_file_dir = os.path.join(base_dir, "temp_image_subset_lists", out_file)
		with open(out_file_dir, 'w+') as out_file:
			for file in batched:
				out_file.write(file)
				out_file.write('\n')
		counter += 1
		print(f"splitting at {batch_start} to {batch_end}, batch number {counter}")
		print("batch number at", counter)


if ((counter*batch_size)+1) >= image_length:
	remainder = image_length-(counter*batch_size)
	if remainder!=0:
		batch_start_last = image_length-remainder
		batched_end = file_list[batch_start_last:image_length]
		print("batch_start_last", batch_start_last, "image_length", image_length)
		out_file = f"file_list_batch_{counter}.txt"
		out_file_dir = os.path.join(base_dir, "temp_image_subset_lists", out_file)
		with open(out_file_dir, 'w+') as out_file:
			for file in batched_end:
				out_file.write(file)
				out_file.write('\n')
		counter += 1
		print("number of batches equal:", counter-1)
	else:
		print("number of batches equal:", counter)

#Carrying out functions on batches

batch_list = glob.glob(os.path.join(base_dir, "temp_image_subset_lists/*.txt"))


f = open("output.txt", "a")
counter = 0
for count, batch in enumerate(batch_list):
	print(f"starting {batch}", file=f)
	with open (batch, newline='') as img_batch_file:
		images_to_copy_list = img_batch_file.read().split('\n')
		images_to_copy_list = images_to_copy_list[:-1]
		for images in images_to_copy_list:
			print(images)
			shutil.copy(images, temp_image_subset, follow_symlinks=True) #set images as filename and os.path.join to create shutil
		linking_functions.predict_leaf(base_dir, "main")
		print(f"{batch} leaf dimension predicted", file=f)
		linking_functions.extract_leaves(base_dir, "main", "Y")
		print(f"{batch} leaf cropped", file=f)
		linking_functions.predict_from_classifier(base_dir, "main")
		print(f"{batch} classifier predicted", file=f)
		linking_functions.removing_files (base_dir) 
		print(f"{batch} files removed", file=f)
		linking_functions.running_R.py (base_dir)
		print(f"{batch} leaf traits extracted", file=f)
		list_remove = glob.glob(os.path.join("base_dir/temp*"), recursive=True)
		for filepath in list_remove:
			os.remove(filepath)
		print(f"{batch} removing temporary files", file=f)
	print(f"completed {batch}", file=f)
f.close()