#!/bin/bash
#mkdir model/
#mkdir model/classifier 
#echo "Making directories: /model, /model/d2, /model/classifier"

conda activate pytorch
touch log.txt
echo "Made log.file" >> log.txt

base_dir=`pwd`
echo "Base directory set as" $base_dir >> log.txt

curr_date=`date`
echo $curr_date >> log.txt

touch classifier_results_test.csv final_results.csv joined_final_results.csv test_duplicates_out.csv
echo "Making file: /classifier_results_test.csv, /final_results.csv, /joined_final_results.csv, /test_duplicates_out.csv" >> log.txt
python /home/botml/code/dev/main_loop/adding_header.py $base_dir

mkdir temp_image_subset image_subset_lists
echo "Making folder: temp_image_subset, image_subset_lists" >> log.txt

mkdir classifier_training_data

python /home/botml/code/dev/main_loop/batching_files.py $base_dir 1000 >> log.txt
batch_list=`ls -d $PWD/image_subset_lists/*`

echo "batches are" ${batch_list}

for batch in $batch_list; do 
	echo "starting" ${batch} >> log.txt
	now_time=`date`
	echo $now_time >> log.txt
	conda activate pytorch
	touch temp_filter.csv 
	touch temp_classifier_results_test.csv
	echo "Making files: /temp_classifier_results_test.csv, /temp_filter.csv" 
	
	mkdir temp_image_subset 
	echo "Making directories: /temp_image_subset"

	mkdir temp_pred 
	mkdir temp_pred_leaf
	mkdir temp_pred_leaf/subf 
	echo "Making directories: /temp_pred, /temp_pred_leaf"
	
	for i in $(cat ${batch}); do echo copying ${i} >> log.txt; image=`basename ${i}`; ln -s ${i} temp_image_subset/$image; done 
	echo "images copied"
  
	python /home/botml/code/dev/main_loop/predict_leaf.py $base_dir "main"
	echo ${batch} "leaf dimension predicted" >> log.txt
	echo "leaves predicted"
	python /home/botml/code/dev/main_loop/extract_leaves_mt.py $base_dir "main" "Y"
	echo ${batch} "leaves cropped" >> log.txt
	echo "leaves cropped"
  python /home/botml/code/dev/main_loop/removing_duplicates.py $base_dir
  echo ${batch} "duplicates tracked" >> log.txt
  echo "duplicates tracked"
	python /home/botml/code/dev/main_loop/predict_from_classifier.py $base_dir "main"
	echo ${batch} "classifier predicted" >> log.txt
	echo "classifier classed"  
	python /home/botml/code/dev/main_loop/removing_files.py $base_dir
	conda activate MLpredictions 
	python /home/botml/code/dev/main_loop/running_R_mt.py $base_dir
  echo ${batch} "leaves traited" >> log.txt 
	rm -r temp*
	echo ${batch} "removed temporary files" >> log.txt
	echo ${batch} "completed!" >> log.txt
	echo "batch completed :)"
done

conda activate pytorch
python /home/botml/code/dev/main_loop/left_join_results.py $base_dir --fields 'id' --fields 'decimalLatitude' --fields 'decimalLongitude' --fields 'genus' --fields 'specificEpithet' 2>> log.txt
echo "merged results with meta" >> log.txt
