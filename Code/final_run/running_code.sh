#!/bin/bash
#mkdir model/
#mkdir model/classifier 
#echo "Making directories: /model, /model/d2, /model/classifier"

#Activating the conda environment
conda activate pytorch

# Create a log file to record the progress of the script
touch log.txt
echo "Made log.file" >> log.txt

# Set the base directory to the current working directory and record it in the log file
base_dir=`pwd`
echo "Base directory set as" $base_dir >> log.txt

#Append date to the log
curr_date=`date`
echo $curr_date >> log.txt

# Create several files that will be used to store the results of the leaf classification model
touch classifier_results_test.csv final_results.csv joined_final_results.csv test_duplicates_out.csv
echo "Making file: /classifier_results_test.csv, /final_results.csv, /joined_final_results.csv, /test_duplicates_out.csv" >> log.txt

# Call a Python script to add headers to the result files
python /home/botml/code/dev/main_loop/adding_header.py $base_dir

# Create two directories to store temporary files
mkdir temp_image_subset image_subset_lists
echo "Making folder: temp_image_subset, image_subset_lists" >> log.txt

# Create a directory to store the training data for the leaf classification model
mkdir classifier_training_data

# Call a Python script to batch the input files into groups of 1000 and store the batch list in a variable
python /home/botml/code/dev/main_loop/batching_files.py $base_dir 1000 >> log.txt
batch_list=`ls -d $PWD/image_subset_lists/*`

echo "batches are" ${batch_list}

# Iterate over each batch in the batch list
for batch in $batch_list; do 
	# Documentation of batch and date
  echo "starting" ${batch} >> log.txt
	now_time=`date`
	echo $now_time >> log.txt
  
  # Activating conda environment
	conda activate pytorch
	
  # Create two temporary files to store the results of the leaf classification model
  touch temp_filter.csv 
	touch temp_classifier_results_test.csv
	echo "Making files: /temp_classifier_results_test.csv, /temp_filter.csv" 
	
  # Create a directory to store temporary images
	mkdir temp_image_subset 
	echo "Making directories: /temp_image_subset"

  # Create three directories to store temporary results from the leaf classification model
	mkdir temp_pred 
	mkdir temp_pred 
	mkdir temp_pred_leaf
	mkdir temp_pred_leaf/subf 
	echo "Making directories: /temp_pred, /temp_pred_leaf"
	
  # Loop over each image in the batch and copy it to the temporary image directory
	for i in $(cat ${batch}); do echo copying ${i} >> log.txt; image=`basename ${i}`; ln -s ${i} temp_image_subset/$image; done 
	echo "images copied"
  
  # Call a Python script to predict the leaf images from the leaf segmentation model
	python /home/botml/code/dev/main_loop/predict_leaf.py $base_dir "main"
	echo ${batch} "leaf dimension predicted" >> log.txt
	echo "leaves predicted"
  # Call a Python script to crop the leaf images
	python /home/botml/code/dev/main_loop/extract_leaves_mt.py $base_dir "main" "Y"
	echo ${batch} "leaves cropped" >> log.txt
	echo "leaves cropped"
  # Call a Python script to track duplicate leaf images from leaf segmentation model
  python /home/botml/code/dev/main_loop/removing_duplicates.py $base_dir
  echo ${batch} "duplicates tracked" >> log.txt
  echo "duplicates tracked"
  # Call a Python script to predict the class of the leaf images
	python /home/botml/code/dev/main_loop/predict_from_classifier.py $base_dir "main"
	echo ${batch} "classifier predicted" >> log.txt
	echo "classifier classed"  
  # Call a Python script to remove invalid leaves
	python /home/botml/code/dev/main_loop/removing_files.py $base_dir
	# Activate conda environment for trait extraction
  conda activate MLpredictions 
  # Call a Python script that calls the trait-extraction R script using multithreads
	python /home/botml/code/dev/main_loop/running_R_mt.py $base_dir
  echo ${batch} "leaves traited" >> log.txt 
	# Removing temporary files
  rm -r temp*
	echo ${batch} "removed temporary files" >> log.txt
	echo ${batch} "completed!" >> log.txt
	echo "batch completed :)"
done

# After all batches are completed, the final output is left-merged with the herbarium meta data by their UID
conda activate pytorch
python /home/botml/code/dev/main_loop/left_join_results.py $base_dir --fields 'id' --fields 'decimalLatitude' --fields 'decimalLongitude' --fields 'genus' --fields 'specificEpithet' 2>> log.txt
echo "merged results with meta" >> log.txt
