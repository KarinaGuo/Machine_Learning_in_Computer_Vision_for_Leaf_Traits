# functions to perform jobs for machine learning of herbarium images

def train_model(training_path, training_name, out_dir, out_yaml, in_yaml, in_weights, in_yaml_zoo, in_weights_zoo, ims_per_batch, base_lr, max_iter, num_classes):
   '''
   Trains a model using Detectron2. This function expects a highly
   prescribed directory and file structure. 

   training_path (str): a valid path that contains training data [e.g. "/srv/scratch/cornwell/syzygium/image_data/"]
   training_name (str): a name for the training dataset [e.g. "train"]  

   The directory nominated in training_path needs to contain coco format label data 
   named with training_name [e.g. train.json] and a directory containing the images [e.g. train/*.jpg] 

   out_dir       (str): a name for an output directory [e.g. "model_v1_out"]
   out_yaml      (str): a name for a file to store the model config [e.g. "model_v1.yaml"]
   in_yaml       (str): a name for the starting config [e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
                        This can be from model zoo, or not... 
   in_weights    (str): a name for the starting weights [e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
                        This can be from model zoo, or not...                        
   in_yaml_zoo    (bool): yaml file from model zoo
   in_weights_zoo (bool): weights from model zoo
   ims_per_batch  (int): images per batch (e.g 4)
   base_lr        (int): initial learning rate (e.g. 0.00025)
   max_iter       (int): max iterations (e.g. 3000)
   '''
   import torch, torchvision
   import numpy as np
   import os, json, cv2, random
   # import some common detectron2 utilities
   import detectron2
   from detectron2.data.datasets import register_coco_instances
   from detectron2.utils.logger import setup_logger
   from detectron2 import model_zoo
   from detectron2.engine import DefaultTrainer
   from detectron2.engine import DefaultPredictor
   from detectron2.config import get_cfg
   setup_logger()
   os.chdir(training_path)
   training_json = f"{training_name}.json"
   register_coco_instances(training_name, {}, training_json, training_name)
   cfg = get_cfg()
   if in_yaml_zoo :
      cfg.merge_from_file(model_zoo.get_config_file(in_yaml))
   else :
      cfg.merge_from_file(in_yaml)
   cfg.DATASETS.TRAIN = (training_name,)
   cfg.DATASETS.TEST = ()
   cfg.DATALOADER.NUM_WORKERS = 2
   if in_weights_zoo :
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(in_weights)
   else :
      cfg.MODEL.WEIGHTS = in_weights
   cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
   cfg.SOLVER.BASE_LR       = base_lr
   cfg.SOLVER.MAX_ITER      = max_iter
   cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
   cfg.OUTPUT_DIR = out_dir
   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
   trainer = DefaultTrainer(cfg)
   trainer.resume_or_load(resume=False)
   trainer.train()
   out_config_file=os.path.join(cfg.OUTPUT_DIR, out_yaml)
   f = open(out_config_file, 'w')
   f.write(cfg.dump())
   f.close()


def predict_in_directory(img_dir, out_name, fext, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7):
   '''
   img_dir (str):  directory containing images for prediction ["/srv/scratch/cornwell/syzygium/image_data/test/"] 
   out_name (str): name for .npy file for predictions ["/srv/scratch/cornwell/syzygium/image_data/model_v1_out_test"]
   fext (str):     file extension for images ["png"]
   yaml_file (str): yaml for model ["/srv/scratch/cornwell/syzygium/image_data/model_v1_out/model_v1.yaml"]
   weights_file (str): weights for model ["/srv/scratch/cornwell/syzygium/image_data/model_v1_out/model_final.pth"]
   yaml_zoo (bool):  yaml from zoo [False]
   weights_zoo (bool): = weights from zoo [False]
   score_thresh = 0.7
   '''
   import torch, torchvision
   import numpy as np
   import os, json, cv2, random
   # import some common detectron2 utilities
   import detectron2
   from detectron2.data.datasets import register_coco_instances
   from detectron2.utils.logger import setup_logger
   from detectron2 import model_zoo
   from detectron2.engine import DefaultTrainer
   from detectron2.engine import DefaultPredictor
   from detectron2.config import get_cfg
   from detectron2.utils.visualizer import Visualizer
   from detectron2.data import MetadataCatalog, DatasetCatalog
   setup_logger()
   new_cfg = get_cfg()
   if yaml_zoo :
      new_cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
   else :
      new_cfg.merge_from_file(yaml_file)
   new_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
   new_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
   if weights_zoo :
      new_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
   else :
      new_cfg.MODEL.WEIGHTS = weights_file
   predictor = DefaultPredictor(new_cfg)
   os.chdir(img_dir)
   images=[f for f in os.listdir(img_dir) if f.endswith('.'+fext)]
   predictions = {}
   prediction_file = out_name + '_predictions.npy'
   for i in range(len(images)):
      i_file_name = images[i]
      i_im = cv2.imread(i_file_name)
      i_outputs = predictor(i_im)
      i_outputs_cpu = i_outputs["instances"].to("cpu")
      #prediction = {
      #   i_file_name : i_outputs_cpu,
      #}
      #predictions.append(prediction)
      predictions[ i_file_name ]=i_outputs_cpu
   np.save(prediction_file, predictions)


def open_predictions(model_predictions_file):
   import torch, torchvision
   import numpy as np
   import detectron2
   np_load_old = np.load
   np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
   model_predictions_in = np.load(model_predictions_file)
   np.load = np_load_old
   model_predictions=model_predictions_in.item()
   # model_predictions['84.png'].pred_masks.numpy()
   return model_predictions


def compare_masks(m1, m2):
   import numpy as np
   s1 = np.sum(m1)
   s2 = np.sum(m2)
   inx = np.sum(np.logical_and(m1,m2))
   unx = np.sum(np.logical_or(m1,m2))
   iou = inx / unx
   return s1, s2, inx, unx, iou


def match_groundtruth_prediction(groundtruth_name, groundtruth_path, model_predictions_file, thresh_iou, matches_out_file):
   import numpy as np
   import os, json, cv2, random
   import torch, torchvision
   import detectron2
   import pandas as pd
   from detectron2.data.datasets import register_coco_instances
   from detectron2.data import MetadataCatalog, DatasetCatalog
   os.chdir(groundtruth_path)
   groundtruth_json = f"{groundtruth_name}.json"
   register_coco_instances(groundtruth_name, {}, groundtruth_json, groundtruth_name)
   groundtruth_metadata = MetadataCatalog.get(groundtruth_name)
   groundtruth_dicts    = DatasetCatalog.get(groundtruth_name)
   number_test_files = len(groundtruth_dicts)
   model_predictions = open_predictions(model_predictions_file)
   matches = []
   for f in range(number_test_files):
      gt_file_name  = groundtruth_dicts[f]['file_name']
      pr_file_name  = os.path.basename(gt_file_name)
      image_data    = cv2.imread(gt_file_name)
      id_shape_0    = image_data.shape[:2][0]
      id_shape_1    = image_data.shape[:2][1]
      pr_num_segs   = len(model_predictions[pr_file_name].pred_masks.numpy())
      gt_num_segs   = len(groundtruth_dicts[f]['annotations'])
      for ind_gt in range(gt_num_segs):
         gt_cat_id = groundtruth_dicts[f]['annotations'][ind_gt]['category_id']
         gt_poly   = groundtruth_dicts[f]['annotations'][ind_gt]['segmentation']
         gt_mask   = detectron2.structures.masks.polygons_to_bitmask(gt_poly,id_shape_0,id_shape_1)
         for ind_pr in range(pr_num_segs): 
            pr_cat_id = model_predictions[pr_file_name].pred_classes.numpy()[ind_pr]
            pr_mask   = model_predictions[pr_file_name].pred_masks.numpy()[ind_pr]
            gt_size_px, pr_size_px, mask_inter, mask_union, mask_iou = compare_masks(gt_mask, pr_mask)
            if mask_iou > thresh_iou:
               dout = {
                   'file_name' : pr_file_name,
                   'ind_gt' : ind_gt,
                   'gt_cat_id' : gt_cat_id,
                   'gt_size_px' : gt_size_px,
                   'ind_pr' : ind_pr,
                   'pr_cat_id' : pr_cat_id,
                   'pr_size_px' : pr_size_px,
                   'mask_iou' : mask_iou
               }
               matches.append(dout)
   matches_df = pd.DataFrame(matches)
   matches_df.to_csv(matches_out_file, index=False) 



def predictions_summary(model_predictions_file, instances_out_file):
   import numpy as np
   import os, json, cv2, random
   import torch, torchvision
   import detectron2
   import pandas as pd
   model_predictions = open_predictions(model_predictions_file)
   number_test_files = len(model_predictions)
   instances = []
   for f in range(number_test_files):
      pr_file_name  = list(model_predictions.keys())[f]
      pr_num_segs   = len(model_predictions[pr_file_name].pred_masks.numpy())
      for ind_pr in range(pr_num_segs): 
         pr_cat_id  = model_predictions[pr_file_name].pred_classes.numpy()[ind_pr]
         pr_mask    = model_predictions[pr_file_name].pred_masks.numpy()[ind_pr]
         pr_score   = model_predictions[pr_file_name].scores.numpy()[ind_pr]
         pr_size_px = np.sum(pr_mask)
         dout = {
                   'file_name' : pr_file_name,
                   'ind_pr' : ind_pr,
                   'pr_cat_id' : pr_cat_id,
                   'pr_score' : pr_score,
                   'pr_size_px' : pr_size_px
                }
         instances.append(dout)
   instances_df = pd.DataFrame(instances)
   instances_df.to_csv(instances_out_file, index=False) 



#https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
def generic_classifier(data_dir, val_ratio, num_epochs, model_out):
   import matplotlib.pyplot as plt
   import numpy as np
   import torch
   from torch import nn
   from torch import optim
   import torch.nn.functional as F
   from torchvision import datasets, transforms, models
   def load_split_train_test(datadir, valid_size = val_ratio):
      #train_transforms = transforms.Compose([transforms.Resize(224),
      #                                   transforms.ToTensor(),
      #                                   ])    
      #test_transforms = transforms.Compose([transforms.Resize(224),
      #                                  transforms.ToTensor(),
      #                                  ])    
      train_transforms = transforms.Compose([transforms.Resize((500,500)),
                                         transforms.ToTensor(),
                                         ])    
      test_transforms = transforms.Compose([transforms.Resize((500,500)),
                                        transforms.ToTensor(),
                                        ])    
      train_data = datasets.ImageFolder(datadir,       
                      transform=train_transforms)
      test_data = datasets.ImageFolder(datadir,
                      transform=test_transforms)    
      num_train = len(train_data)
      indices = list(range(num_train))
      split = int(np.floor(valid_size * num_train))
      np.random.shuffle(indices)
      from torch.utils.data.sampler import SubsetRandomSampler
      train_idx, test_idx = indices[split:], indices[:split]
      train_sampler = SubsetRandomSampler(train_idx)
      test_sampler = SubsetRandomSampler(test_idx)
      trainloader = torch.utils.data.DataLoader(train_data,
                     sampler=train_sampler, batch_size=24)
      testloader = torch.utils.data.DataLoader(test_data,
                     sampler=test_sampler, batch_size=12)
      return trainloader, testloader
   trainloader, testloader = load_split_train_test(data_dir, val_ratio)
   device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
   model = models.resnet50(pretrained=True)
   for param in model.parameters():
      param.requires_grad = False
   model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))
   criterion = nn.NLLLoss()
   optimizer = optim.Adam(model.fc.parameters(), lr=0.0002)
   model.to(device)
   epochs = num_epochs
   steps = 0
   running_loss = 0
   print_every = 1
   train_losses, test_losses = [], []
   for epoch in range(epochs):
      for inputs, labels in trainloader:
         steps += 1
         inputs, labels = inputs.to(device), labels.to(device)
         optimizer.zero_grad()
         logps = model.forward(inputs)
         loss = criterion(logps, labels)
         loss.backward()
         optimizer.step()
         running_loss += loss.item()
         if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
               for inputs, labels in testloader:
                   inputs, labels = inputs.to(device), labels.to(device)
                   logps = model.forward(inputs)
                   batch_loss = criterion(logps, labels)
                   test_loss += batch_loss.item()
                   ps = torch.exp(logps)
                   top_p, top_class = ps.topk(1, dim=1)
                   equals = top_class == labels.view(*top_class.shape)
                   accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
   torch.save(model, model_out)


def visualize_predictions(img_dir, out_dir, fext, training_path, training_name, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7, s1=10, s2=14):
   '''
   img_dir (str):  directory containing images for prediction ["/srv/scratch/cornwell/syzygium/image_data/test/"] 
   out_dir (str):  directory for predictions ["/srv/scratch/cornwell/syzygium/image_data/pred_img/"]
   fext (str):     file extension for images ["png"]
   yaml_file (str): yaml for model ["/srv/scratch/cornwell/syzygium/image_data/model_v1_out/model_v1.yaml"]
   weights_file (str): weights for model ["/srv/scratch/cornwell/syzygium/image_data/model_v1_out/model_final.pth"]
   yaml_zoo (bool):  yaml from zoo [False]
   weights_zoo (bool): = weights from zoo [False]
   score_thresh = 0.7
   '''
   import torch, torchvision
   import numpy as np
   import os, json, cv2, random
   # import some common detectron2 utilities
   import detectron2
   from detectron2.data.datasets import register_coco_instances
   from detectron2.utils.logger import setup_logger
   from detectron2 import model_zoo
   from detectron2.engine import DefaultTrainer
   from detectron2.engine import DefaultPredictor
   from detectron2.config import get_cfg
   from detectron2.utils.visualizer import Visualizer
   from detectron2.data import MetadataCatalog, DatasetCatalog
   import matplotlib.pyplot as plt
   setup_logger()
   os.chdir(training_path)
   training_json = f"{training_name}.json"
   register_coco_instances(training_name, {}, training_json, training_name)
   training_metadata = MetadataCatalog.get(training_name)
   new_cfg = get_cfg()
   if yaml_zoo :
      new_cfg.merge_from_file(model_zoo.get_config_file(yaml_file))
   else :
      new_cfg.merge_from_file(yaml_file)
   new_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh 
   new_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
   if weights_zoo :
      new_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
   else :
      new_cfg.MODEL.WEIGHTS = weights_file
   predictor = DefaultPredictor(new_cfg)
   os.chdir(img_dir)
   images=[f for f in os.listdir(img_dir) if f.endswith('.'+fext)]
   for i in range(len(images)):
      i_file_name = images[i]
      i_im = cv2.imread(i_file_name)
      i_outputs = predictor(i_im)
      v = Visualizer(i_im[:, :, ::-1],
         metadata=training_metadata,
         scale=0.8
      )
      v = v.draw_instance_predictions(i_outputs["instances"].to("cpu"))
      h, w, _ = i_im.shape
      s1 = w / 516 * 4
      s2 = h / 516 * 4
      plt.figure(figsize = (s1, s2))
      plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
      out_fil = out_dir + 'predict_' + os.path.basename(i_file_name)
      plt.savefig(out_fil)


def train_val_model(training_path, training_name,validation_name, out_dir, out_yaml, in_yaml, in_weights, in_yaml_zoo, in_weights_zoo, ims_per_batch, base_lr, max_iter, num_classes):
   '''
   Trains a model using Detectron2. This function expects a highly
   prescribed directory and file structure. 

   training_path (str): a valid path that contains training data [e.g. "/srv/scratch/cornwell/syzygium/image_data/"]
   training_name (str): a name for the training dataset [e.g. "train"]  
   validation_name (str): a name for the training dataset [e.g. "val"]
   The directory nominated in training_path needs to contain coco format label data 
   named with training_name [e.g. train.json] and a directory containing the images [e.g. train/*.jpg] 

   out_dir       (str): a name for an output directory [e.g. "model_v1_out"]
   out_yaml      (str): a name for a file to store the model config [e.g. "model_v1.yaml"]
   in_yaml       (str): a name for the starting config [e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
                        This can be from model zoo, or not... 
   in_weights    (str): a name for the starting weights [e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
                        This can be from model zoo, or not...                        
   in_yaml_zoo    (bool): yaml file from model zoo
   in_weights_zoo (bool): weights from model zoo
   ims_per_batch  (int): images per batch (e.g 4)
   base_lr        (int): initial learning rate (e.g. 0.00025)
   max_iter       (int): max iterations (e.g. 3000)
   '''
   import torch, torchvision
   import numpy as np
   import os, json, cv2, random
   # import some common detectron2 utilities
   import detectron2
   from detectron2.data.datasets import register_coco_instances
   from detectron2.utils.logger import setup_logger
   from detectron2 import model_zoo
   from detectron2.engine import DefaultTrainer
   from detectron2.engine import DefaultPredictor
   from detectron2.config import get_cfg
   setup_logger()
   os.chdir(training_path)
   training_json = f"{training_name}.json"
   validation_json = f"{validation_name}.json"
   register_coco_instances(training_name, {}, training_json, training_name)
   register_coco_instances(validation_name, {}, validation_json, validation_name)
   cfg = get_cfg()
   if in_yaml_zoo :
      cfg.merge_from_file(model_zoo.get_config_file(in_yaml))
   else :
      cfg.merge_from_file(in_yaml)
   cfg.DATASETS.TRAIN = (training_name,)
   cfg.DATASETS.TEST = (validation_name,)
   cfg.DATALOADER.NUM_WORKERS = 2
   if in_weights_zoo :
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(in_weights)
   else :
      cfg.MODEL.WEIGHTS = in_weights
   cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
   cfg.SOLVER.BASE_LR       = base_lr
   cfg.SOLVER.MAX_ITER      = max_iter
   cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
   cfg.OUTPUT_DIR = out_dir
   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
   trainer = DefaultTrainer(cfg)
   trainer.resume_or_load(resume=False)
   trainer.train()
   out_config_file=os.path.join(cfg.OUTPUT_DIR, out_yaml)
   f = open(out_config_file, 'w')
   f.write(cfg.dump())
   f.close()

def train_val_model(training_path, training_name,validation_name, out_dir, out_yaml, in_yaml, in_weights, in_yaml_zoo, in_weights_zoo, ims_per_batch, base_lr, max_iter, num_classes):
   '''
   Trains a model using Detectron2. This function expects a highly
   prescribed directory and file structure. 

   training_path (str): a valid path that contains training data [e.g. "/srv/scratch/cornwell/syzygium/image_data/"]
   training_name (str): a name for the training dataset [e.g. "train"]  
   validation_name (str): a name for the training dataset [e.g. "val"]
   The directory nominated in training_path needs to contain coco format label data 
   named with training_name [e.g. train.json] and a directory containing the images [e.g. train/*.jpg] 

   out_dir       (str): a name for an output directory [e.g. "model_v1_out"]
   out_yaml      (str): a name for a file to store the model config [e.g. "model_v1.yaml"]
   in_yaml       (str): a name for the starting config [e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
                        This can be from model zoo, or not... 
   in_weights    (str): a name for the starting weights [e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
                        This can be from model zoo, or not...                        
   in_yaml_zoo    (bool): yaml file from model zoo
   in_weights_zoo (bool): weights from model zoo
   ims_per_batch  (int): images per batch (e.g 4)
   base_lr        (int): initial learning rate (e.g. 0.00025)
   max_iter       (int): max iterations (e.g. 3000)
   '''
   import torch, torchvision
   import numpy as np
   import os, json, cv2, random
   # import some common detectron2 utilities
   import detectron2
   from detectron2.data.datasets import register_coco_instances
   from detectron2.utils.logger import setup_logger
   from detectron2 import model_zoo
   from detectron2.engine import DefaultTrainer
   from detectron2.engine import DefaultPredictor
   from detectron2.config import get_cfg
   from detectron2.evaluation import COCOEvaluator
   class CocoTrainer(DefaultTrainer):
     @classmethod
     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
           os.makedirs("coco_eval", exist_ok=True)
           output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
   setup_logger()
   os.chdir(training_path)
   training_json = f"{training_name}.json"
   validation_json = f"{validation_name}.json"
   register_coco_instances(training_name, {}, training_json, training_name)
   register_coco_instances(validation_name, {}, validation_json, validation_name)
   cfg = get_cfg()
   if in_yaml_zoo :
      cfg.merge_from_file(model_zoo.get_config_file(in_yaml))
   else :
      cfg.merge_from_file(in_yaml)
   cfg.DATASETS.TRAIN = (training_name,)
   cfg.DATASETS.TEST = (validation_name,)
   cfg.DATALOADER.NUM_WORKERS = 2
   if in_weights_zoo :
      cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(in_weights)
   else :
      cfg.MODEL.WEIGHTS = in_weights
   cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
   cfg.SOLVER.BASE_LR       = base_lr
   cfg.SOLVER.MAX_ITER      = max_iter
   cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
   cfg.TEST.EVAL_PERIOD = 100
   cfg.OUTPUT_DIR = out_dir
   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
   #trainer = DefaultTrainer(cfg)
   trainer = CocoTrainer(cfg)
   trainer.resume_or_load(resume=False)
   trainer.train()
   out_config_file=os.path.join(cfg.OUTPUT_DIR, out_yaml)
   f = open(out_config_file, 'w')
   f.write(cfg.dump())
   f.close()



def find_duplicate_predictions(model_predictions_file, duplicates_out_file, iou_thresh):
   import numpy as np
   import os, json, cv2, random
   import torch, torchvision
   import detectron2
   import pandas as pd
   model_predictions = open_predictions(model_predictions_file)
   number_test_files = len(model_predictions)
   print(number_test_files)
   instances = []
   for f in range(number_test_files):
      pr_file_name  = list(model_predictions.keys())[f]
      pr_num_segs   = len(model_predictions[pr_file_name].pred_masks.numpy())
      for ind_pr in range(pr_num_segs):
         pr_cat_id  = model_predictions[pr_file_name].pred_classes.numpy()[ind_pr]
         pr_mask    = model_predictions[pr_file_name].pred_masks.numpy()[ind_pr]
         pr_score   = model_predictions[pr_file_name].scores.numpy()[ind_pr]
         count_gt_thresh = 0
         max_mask_iou = 0
         ind_max_iou = -1
         for comp_pr in range(pr_num_segs):
            if ind_pr != comp_pr: 
              comp_cat_id  = model_predictions[pr_file_name].pred_classes.numpy()[comp_pr]
              comp_mask    = model_predictions[pr_file_name].pred_masks.numpy()[comp_pr]
              comp_score   = model_predictions[pr_file_name].scores.numpy()[comp_pr]
              pr_size_px, comp_size_px, mask_inter, mask_union, mask_iou = compare_masks(pr_mask, comp_mask)
              if mask_iou > iou_thresh:
                count_gt_thresh = count_gt_thresh + 1
                if mask_iou > max_mask_iou:
                  ind_max_iou  = comp_pr
                  max_mask_iou = mask_iou
         dout = {
                   'id' : pr_file_name,
                   'index' : ind_pr,
                   'pr_cat_id' : pr_cat_id,
                   'count_gt_thresh' : count_gt_thresh,
                   'ind_max_iou' : ind_max_iou,
                   'max_mask_iou' : max_mask_iou
                }
         instances.append(dout)
   instances_df = pd.DataFrame(instances)
   instances_df.to_csv(duplicates_out_file, mode='a', index=False, header=False)

