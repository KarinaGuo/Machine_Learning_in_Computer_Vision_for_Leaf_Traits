Hello! This repository includes the key scripts and data files used for my honours thesis. File paths, file names, and such may need to be changed. Let me know if anything is missing or isn't working. 

/Code - includes code used for the model's accuracy predictions, the model creations, and the analysis of the datasets

/Data - includes files used in code

/leaf_BigLeaf_QC - datasets used to analyse the quality control of our machine learning model, as in Supplementary Information D

/Conda Environments - includes the .yml files of the conda environments used to perform the training/testing/use of the machine learning models, and the extraction of traits from the binary masks of the machine learning model predictions 

<b> Below is an in progress documentation used for our workflow... </b>

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

<p><i> Converting file formats to COCO. Where <b>/data</b> is the input directory. <b>--output</b> is the output file. <b>--classes</b> is the desired training label to be included. <b>--polyORbb</b> is whether the annotation is a polygon or a bounding box.</p>

```
python /home/botml/code/py/lm2coco.py /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/data/ --output /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/data.json --classes 'Leaf100' --polyORbb 'poly'

python /home/botml/code/py/lm2coco.py /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/validation/ --output /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/validation.json --classes 'Leaf100' --polyORbb 'poly'

python /home/botml/code/py/lm2coco.py /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/test/ --output /data/botml/leaf_dimension/ELEVEN_DuplTen_ExtSheets/test.json --classes 'Leaf100' --polyORbb 'poly'
```
