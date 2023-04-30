import os, glob, sys
import math
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
parser.add_argument("batch_size", type=int,)
args = parser.parse_args()
base_dir = args.base_dir


file_list = glob.glob('/stage/botml/euc/data/raw/*.jpg') #replace dir with image dir, could possibly do this as a argpars
#file_list = glob.glob('/home/karina/test_images2/*.jpg')
batch_size = math.ceil(len(file_list)/args.batch_size)

image_length = len(file_list)

counter = 1
for img in file_list:
	if ((counter*batch_size)+1) <= image_length:
		batch_start = batch_size * (counter-1)
		batch_end = batch_size * counter
		batched = file_list[batch_start:batch_end]
		out_file = f"file_list_batch_{counter}.txt"
		out_file_dir = os.path.join(args.base_dir, "image_subset_lists", out_file)
		with open(out_file_dir, 'w+') as out_file:
			for file in batched:
				out_file.write(file)
				out_file.write('\n')
		counter += 1
		print(f"splitting at {batch_start} to {batch_end}")
		print("batch number", counter-1)


if ((counter*batch_size)+1) >= image_length:
	print(counter)
	remainder = image_length-((counter-1)*(batch_size))
	if remainder!=0:
		batch_start_last = image_length-remainder
		batched_end = file_list[batch_start_last:image_length]
		print("batch_start_last", batch_start_last, "image_length", image_length)
		out_file = f"file_list_batch_{counter}.txt"
		out_file_dir = os.path.join(args.base_dir, "image_subset_lists", out_file)
		with open(out_file_dir, 'w+') as out_file:
			for file in batched_end:
				out_file.write(file)
				out_file.write('\n')
		counter += 1
		print("number of batches equal:", counter-1)
	else:
		print("number of batches equal:", counter-1)