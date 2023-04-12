<p>This directory includes the scripts used to train, validate and test the two machine learning models. </p>

<h3>Leaf masking model</h3>
<p>The leaf masking model was run with the following scripts
predict_leaf.py generated all summary evaluation metrics (quantitative + qualitative), while predict_leaf_vis.py only generated the qualitative metrics</p>

<ol>
  <li>python /home/botml/code/py/updating_labels.py # Changes the labels of the polygons according to the iteration</li>
  <li>python /home/botml/code/py/cut_focal_box.py # Trims the herbarium sheet images to the bounding box</li>
  <li>Move portion of train labels to validation</li>
  <li>python /home/botml/code/py/lm2coco.py ~/data/ #Converting the training dataset to a COCO dataset</li>
  <li>python /home/botml/code/py/lm2coco.py ~/validation/ #Converting the validation dataset to a COCO dataset</li>
  <li>python /home/botml/code/py/lm2coco.py ~/test/ #Converting the test dataset to a COCO dataset</li>
  <li>Change train_leaf.py, predict_leaf.py, and predict_leaf_vis.py appropriately</li>
  <li>Run train_leaf.py #Training/Validating the model</li>
  <li>Run predict_leaf.py/predict_leaf_vis.py #Testing the model</li>
</ol>

<h3>Leaf classification model</h3>
<ol>
  <li>python extract_leaves_mt.py # Generates the leaf images
  <li>Manually classify leaf images into valid/invalid and move them into their respective /data /validation /test directories </li>
  <li>python /home/botml/code/dev/main_loop/classify_leaves.py # Training the model </li>
  <li>python /home/botml/code/py/predict_from_classifierv2.py # Testing the model </li>
</ol>

<h3>Running the trained models on all herbarium sheets</h3>

```
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
```
