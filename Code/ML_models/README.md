<p>This directory includes the scripts used to train, validate and test the two machine learning models. </p>

<h3>Leaf masking model</h3>
<p>The leaf masking model was run with the following scripts
predict_leaf.py generated all summary evaluation metrics (quantitative + qualitative), while predict_leaf_vis.py only generated the qualitative metrics</p>
<ol>
  <li>python updating_labels.py # Changes the labels of the polygons according to the iteration</li>
  <li>python cut_focal_box.py # Trims the herbarium sheet images to the bounding box</li>
  <li>Move portion of train labels to validation</li>
  <li>python lm2coco.py ~/data/ #Converting the training dataset to a COCO dataset</li>
  <li>python lm2coco.py ~/validation/ #Converting the validation dataset to a COCO dataset</li>
  <li>python lm2coco.py ~/test/ #Converting the test dataset to a COCO dataset</li>
  <li>Change train_leaf.py, predict_leaf.py, and predict_leaf_vis.py appropriately</li>
  <li>python train_leaf.py #Training/Validating the model</li>
  <li>python predict_leaf.py/predict_leaf_vis.py #Testing the model</li>
</ol>

<h3>Leaf classification model</h3>
<ol>
  <li>python extracting_leaves_cropped_iter_v2.py # Generates the leaf images</li>
  <li>Manually classify leaf images into valid/invalid and move them into their respective /data /validation /test directories </li>
  <li>python classify_leaves.py # Training the model </li>
  <li>python predict_from_classifierv2.py # Testing the model </li>
</ol>
