Hello! This repository includes the key scripts and data files used for my honours thesis. File paths, file names, and such may need to be changed. Let me know if anything is missing or isn't working. 

/Code - includes code used for the model's accuracy predictions, the model creations, and the analysis of the datasets

/Data - includes files used in code

/leaf_BigLeaf_QC - datasets used to analyse the quality control of our machine learning model, as in Supplementary Information D

/Conda Environments - includes the .yml files of the conda environments used to perform the training/testing/use of the machine learning models, and the extraction of traits from the binary masks of the machine learning model predictions 

<b> Below is an in-progress documentation used for our workflow... </b>

<h1>Leaf segmentation model</h1>
<h2>Preparing the data</h2>

<p>As this was integrated into a cycle of optimisation, labels of annotated herbarium images for training, validating and testing were changed when two or more labels were merged into one. These annotated images were then trimmed to the bounding box (BB) as stated in the protocol, then converted to a COCO file format.</p>
<p>The conda environment 'labelme' was used for this process</p>

<p><i> Updating labels. Where test_labels is a directory of the initial unchanged annotated images. <b>test_labels_updatedlabs</b> is the output directory. <b>testmap.csv</b> is a dictionary that indicates which old labels map to which new labels. <b>testmap.log</b> is a variable not in use, and is currently an empty placeholder.</i></p>

```
python /home/botml/code/py/updating_labels.py /data/botml/test_labels/ /data/botml/test_labels_updatedlabs/ /data/botml/leaf_dimension/EIGHT_DuplSeven_BS20_ExtTrain/testmap.csv /data/botml/NINE_DuplSeven_BS20_L100_ExtTrain/fb2_vnoUM/testmap.log
```

<p>An example of <b>testmap.csv</b>, where the labels Leaf90 and Leaf100UM are converted to Leaf100</p>

<table>
  <tr>
    <td>Leaf90</td>
    <td>Leaf100UM</td>
  </tr>
  <tr>
    <td>Leaf100</td>
    <td>Leaf100</td>
  </tr>
</table>

<p><i> Trimming the annotated images to the bounding boxes. Where <b>train_labels_updatedlabs</b> is the input directory of the annotated images. <b>train_labels_trimmed</b> is the output directory. <b>--focalbox</b> is the label indicating the bounding box. <b>--classes</b> is the desired classes to be included in the output</i></p>

```
python /home/botml/code/py/cut_focal_box.py /data/botml/train_labels_updatedlabs/ /data/botml/train_labels_trimmed/ --focalbox BB --classes Leaf100
```

<p>A portion of these train labels were then moved to validation (20% of all annotated input data). The input data for training, validating and testing were then converted to a COCO file format.</p>

<p><i> Converting file formats to COCO. Where <b>/data</b> is the input directory. <b>--output</b> is the output file. <b>--classes</b> is the desired training label to be included. <b>--polyORbb</b> is whether the annotation is a polygon or a bounding box.</i><</p>

```
python /home/botml/code/py/lm2coco.py /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/data/ --output /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/data.json --classes 'Leaf100' --polyORbb 'poly'

python /home/botml/code/py/lm2coco.py /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/validation/ --output /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/validation.json --classes 'Leaf100' --polyORbb 'poly'

python /home/botml/code/py/lm2coco.py /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/test/ --output /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/test.json --classes 'Leaf100' --polyORbb 'poly'
```
<p>At this stage, the working directory should include, if they are not present please make an empty directory or download from the data directory in this repository if present:
  <ul>
  <li>/coco_eval</li>
  <li>/data</li>
  <li>/validation</li>
  <li>/test</li>
  <li>/pred</li>
  <li>/code</li>
  <li>data.json</li>
  <li>validation.json</li>
  <li>test.json</li></ul></p>

<h2>Training, validating, and testing the model</h2>
<p>The conda environment 'pytorch' was used for this process</p>
<p>Edit the scripts according to your set up. This would include changing the training_path and the training parameters.</p>
<i><p>Training & validating the model</i>. The following variables are included in the script below, and would likely need to be changed to be relevant.

<p>
  <ul>
    <li><b>training_path</b>: The path to the training data.</li>
    <li><b>training_name</b>: The name of the training directory.</li>
    <li><b>validation_name</b>: The name of the validation directory.</li>
    <li><b>out_dir</b>: The directory where the trained model will be saved.</li>
    <li><b>out_yaml</b>: The name of the YAML file that will be used to save the trained model.</li>
    <li><b>in_yaml</b>: The path to the YAML file that contains the model architecture.</li>
    <li><b>in_weights</b>: The path to the weights file that will be used to initialize the model.</li>
    <li><b>in_yaml_zoo</b>: A boolean value that indicates whether to use the model architecture from the Zoo.</li>
    <li><b>in_weights_zoo</b>: A boolean value that indicates whether to use the weights from the Zoo.</li>
    <li><b>ims_per_batch</b>: The number of images per batch.</li>
    <li><b>base_lr</b>: The base learning rate.</li>
    <li><b>max_iter</b>: The maximum number of iterations.</li>
    <li><b>num_classes</b>: The number of classes.</li>
  </ul>
</p>

```
python train_leaf.py
```
<p>The trained model will be saved in the /model directory</p>

<i><p>Testing the model</i>. The following variables are included in the script below, and may need to be changed to be relevant. Following this, quantitative evaluation metrics were calculated using machine_accuracy_process_v2.R.</p>

<ul>
  <li><b>img_dir</b>: The directory where the test images are located.</li>
  <li><b>out_dir</b>: The directory where the predictions will be saved.</li>
  <li><b>fext</b>: The file extension of the test images.</li>
  <li><b>training_path</b>: The path to the training data.</li>
  <li><b>training_name</b>: The name of the training directory.</li>
  <li><b>yaml_file</b>: The path to the YAML file that contains the model architecture.</li>
  <li><b>weights_file</b>: The path to the weights file that will be used to initialize the model.</li>
  <li><b>yaml_zoo</b>: A boolean value that indicates whether to use the model architecture from the Zoo.</li>
  <li><b>weights_zoo</b>: A boolean value that indicates whether to use the weights from the Zoo.</li>
  <li><b>num_classes</b>: The number of classes.</li>
  <li><b>score_thresh</b>: The score threshold for determining whether a prediction is positive.</li>
  <li><b>s1</b>: The first scale factor for resizing the images.</li>
  <li><b>s2</b>: The second scale factor for resizing the images.</li>
  <li><b>groundtruth_name</b>: The name of the ground truth directory.</li>
  <li><b>model_predictions_file</b>: The path to the file that contains the model predictions.</li>
  <li><b>thresh_iou</b>: The intersection-over-union threshold for determining whether a prediction matches a ground truth instance.</li>
  <li><b>matches_out_file</b>: The path to the file where the matches will be saved.</li>
  <li><b>instances_out_file</b>: The path to the file where the instance summaries will be saved.</li>
</ul>

<p>The following functions are used in the Python script:</p>

<ul>
  <li><b>model_tools.visualize_predictions()</b>: This function visualizes the predictions for the test images.</li>
  <li><b>model_tools.predict_in_directory()</b>: This function predicts the labels for the test images.</li>
  <li><b>model_tools.match_groundtruth_prediction()</b>: This function matches the model predictions to the ground truth instances.</li>
  <li><b>model_tools.summarize_predictions()</b>: This function summarizes the model predictions.</li>
</ul>

```
python predict_leaf.py
```

<p>Or alternatively the below can be run on images that have not been annotated. This creates the qualitative metrics only, skipping the quantitative evaluation metrics.</p>

```
python predict_leaf_vis.py
```

<h1>Leaf classification model</h1>
<h2>Preparing the data</h2>
<p>First, the leaf segmentation model is used on a dataset to create the leaf masks required for the training & validation steps of this model. To do this, the predict_leaf_vis.py script is run, which creates a numpy file that includes the binary masks of each leaf per image. These binary masks are then extracted, resized, padded, recoloured and a connected component analysis with OTSU thresholding applied. This is done with the script below</p>

<p><i>Extracting leaf masks from the leaf segmentation model predictions.</i> Where the structure is as such 
  <ul>
    <li><b>_predictions.npy</b>: Numpy file from the output of the segmentation mask</li>
    <li><b>/testing_images</b>: Images used on the leaf segmentation model, to generate the numpy file above</li>
    <li><b>/classifier_training_testdata</b>: Output file of leaf images</li>
  </ul>
</p>

```
python /data/botml/leaf_dimension_classifier/code/extracting_leaves_cropped_iter_v2.py "/data/botml/leaf_dimension_classifier/testing_images_pred/_predictions.npy" "/data/botml/leaf_dimension_classifier/testing_images/" "/data/botml/leaf_dimension_classifier/input_classifier_data/"
```

<p>These images are then manually separated into 'valid' and 'invalid' classes in a separate directory as /classed_training_data/N and /classed_training_data/Y. At this stage, the working directory should include, if they are not present please make an empty directory or download from the data directory in this repository if present:
  
<ul>
  <li><b>/code</b>: The directory of relevant code</li>
  <li><b>/model/classifier</b>: Directory where the final model is stored</li>
  <li><b>/pred_leaf</b>: Full images of used to create the training dataset</li>
  <li><b>/input_classifier_data</b>: The cropped unclassed images used for training/validating, from the predictions of the leaf segmentation model</li>
  <li><b>/classed_training_data</b>: The cropped classed images used for training/validating, from the predictions of the leaf segmentation model</li>
  <li><b>/classifier_training_testdata</b>: The cropped classed images used for testing, from the predictions of the leaf segmentation model</li>
  <li><b>classifier_results_test.csv</b>: A .csv file that will include the predictions of the model</li>
</ul>

<h2>Training, validating, and testing the model</h2>
<p>Edit the scripts according to your set up. This would include changing the training_path and the training parameters.</p>
<i><p>Training & validating the model</i>. The following variables are included in the script below, and would likely need to be changed to be relevant.
<ul>
  <li><b>data_dir</b>: The directory where the classified training data is located.</li>
  <li><b>val_ratio</b>: The fraction of the training data to use for validation.</li>
  <li><b>num_epochs</b>: The number of epochs to train the model for.</li>
  <li><b>model_out</b>: The path to the file where the trained model will be saved.</li>
</ul></p>

```
python classify_leaves.py
```
<p>The final model is deposited into /model/classifier</p>
<p>The previous steps for preparing the data is repeated for the test leaf images and is placed in /classifier_training_testdata.</p>
<i><p>Testing the model</i>. The following variables are included in the script below, and would likely need to be changed to be relevant.
<ul>
  <li><b>train_dir</b>: The directory where the classified training data is located.</li>
  <li><b>data_dir</b>: The directory where the test data is located.</li>
  <li><b>model_file</b>: The path to the file where the trained model will be saved.</li>
  <li><b>out_dir</b>: The path to the file where the results will be saved.</li>
</ul></p>

```
python predict_from_classifierv2.py
```

<p>Evaluation metrics were calculated using machine_accuracy_process_v2.R after manually editing the output of the classifier.</p>

<h1>Extracting traits</h1>
<p>Traits were extracted using an R script leaf_dimension_calculations.R where each leaf mask is fed in as an argument. This was integrated into the final loop we used over the entire herbarium dataset. Please refer to below for running the leaf trait extraction script.</p>

<h1>Using the final models on the entire dataset</h1>
<p>The final model outputs were then moved to the directories /model/d2, /model/classifier, nesting in the main working directory of the final run.</p>
<p>The final run was then executed using the bash script running_code.sh in a working directory, it calls tailored codes that can be found in this repository under /data/final_run. This script performs the following operations:

<ol>
  <li>Copies the images to a temporary directory.</li>
  <li>Predicts the dimensions of the leaves in the images.</li>
  <li>Crops the leaves from the images.</li>
  <li>Tracks duplicates.</li>
  <li>Predicts the class of the leaves in the images.</li>
  <li>Removes temporary files.</li>
  <li>Executes R code to generate traits for the leaves.</li>
  <li>Deletes the temporary files.</li>
  <li>The script then merges the outcomes with the metadata and displays a message indicating that the script has finished executing.</li>
</ol>
</p>
