# For parsing command line arguments
import argparse
import os
import sys
import time
# Handle folder and directories platform independently
from pathlib import Path
import cv2
import torch
# Backend for Cuda Deep Neural Network Library
import torch.backends.cudnn as cudnn

# Allows duplicate libraried for OpenMP, KMP=(Kernel Math Parallelism)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Add this path to the list of searching modules without changing base path, at index 0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "yolov5"))
# Loading yolo v5 model
from yolov5.models.experimental import attempt_load
# Loading Image/video data for yolo v5 model
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
     xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, scale_coords
# Plot bounding boxes in yolo v5
from yolov5.utils.plots import colors, plot_one_box
# Custom Functions to visualize bounding boxes
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

# Parsing configurations for DeepSort
from deep_sort_pytorch.utils.parser import get_config
# DeepSort Object tracker
from deep_sort_pytorch.deep_sort import DeepSort

from graphs import bbox_rel, draw_boxes


# Function for detecting objects using YOLO v5 and applying DeepSort for tracking
@torch.no_grad()
def detect(weigts = 'yolo5s.pt',
          source = 'yolov5/data/images',
          imgshz = 640,               # Inference size(pixels)
          conf_thres = 0.25,          # Confidence threshold
          iou_thres = 0.45,           # NMS Intersection over Union threshold
          max_det = 1000,             # Maximum detections per image
          device = '',                # Cuda device
          view_img = False, 
          save_txt = False,           # Save results to *.txt
          save_conf = False,          # Save confidence in --save-txt labels
          nosave = False,             # Do not save images/videos
          classes = None,             # Filter by class:0, 1, 2
          agnostic_nms = False,       # Uses Class-Aware Non max supression
          augment = False, 
          update = False,             # Update all models
          projects = 'runs/detect',   # Save results to project/name
          name = 'exp',               # Save results to project/name
          exist_ok = False,           
          line_thickness = 3,         # Bounding box thickness(pixels)
          hide_labels = False,
          hide_conf = False,
          half = False,               # Half precision for inference is used
          config_deepsort = 'deep_sort_pytorch/configs/deep_sort/yaml'):
     # Saves inference image
     save_img = not nosave and not source.endswith('.txt')
     # Check if the source is a webcam, a file wih .txt extension or a URL stream
     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
          ('rtsp://', 'rtmp://', 'http://', 'https://'))

     # Initilaize DeepSort by loading configurations fro a specified file
     cfg = cgf.get_config()
     cfg.merge_from_file(opt.config_deepsort)
     deepsort = DeepSort(
          # Path to ReID (Re-identification) model checkpoint
          cfg.DEEPSORT.REID_CKPT,
          # Maximum allowable distance b/w object embedding for matching
          max_dist = cfg.DEEPSORT.MAX_DIST,
          # Minimum confidence score for object detection
          min_confidence = cfg.DEEPSORT.MIN_CONFIDENCE,
          # Max overlap allowed for NMS
          nms_max_overlap = cfg.DEEPSORT.NMS_MAX_OVERLAP,
          # Maximum IoU distance
          max_iou_distance = cfg.DEEPSORT.MAX_IOU_DISTANCE,
          # Maximum age of an object track
          max_age = cfg.DEEPSORT.MAX_AGE,
          # No. of consecutive frames to activate the tracker
          n_init = cfg.DEEPSORT.N_INIT,
          # Siz of appearance descriptor distance metric cache
          nn_budget = cfg.DEEPSORT.NN_BUDGET,
          use_cuda = True
     )
     # Increment the path for saving results, creating a new directory if needed
     save_dir = increment_path(Path(project)/name, exist_ok=exist_ok)

     # Create 'labels' directory within 'save_dir' if save_txt is True, otherwise create 'save_dir'
     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

     # Initialize up logging
     set_logging()

     device = select_device(device)

     # Check if half precision is supported and applicable for current device
     # wiLL be disabled if device=cpu else enabled
     half &=device.type!='cpu'
     
     # will load full precision(Float 32 ) model
     model = attempt_load(weights=weigts, map_location=device)

     # Determine the stride of the model for scaling purpose
     stride = int(model.stride.max())

     # Check and adjust the image size if necessary based on model's stride
     imgsz = check_img_size(img_size=imgsz, s=stride)

     # Get class names from the model
     names = model.module.names if hasattr(obj=model, name='module') else model.names

     # Convert the model to FP16(half precision), if applicable
     if half:
          model.half()

     # Second-stage classifier
     classifier = False
     if classify:
          modelc = load_classifier(name='resnet101', n=2) # n is the number of classes
          modelc.load_state_dict(torch.load(f='weights/resnet101.pt', map_location=device)['model']).to(device).eval()

     # Set dataloader
     vid_path, vid_writer = None, None

     if webcam:
          # Check if viewer wants to view image during processing
          view_img = check_imshow()

          # Speed up constant image size inference if using CUDA
          cudnn.benchmark = True

          # Load a video stream if the source is a webcam
          dataset = LoadStreams(sources=source, img_size=imgsz, stride=stride)

     else:
          # Load images or a video file as the dataset
          dataset = LoadImages(path=source, img_size=imgszm, stride=stride)

     # Run inference once an empty tensor tp initilize the model
     # Useful for CUDA-based devices to set up model parameters
     if device.type != 'cpu':
          # size=(batch-size, #channels, height, width)
          model(torch.zeros(size=(1, 3, imgsz, imgsz)).to(device).type_as(next(model.parameters())))
     
     t0 = time.time()#Current time is seconds since epoch

     # Iterate through frames and relevant data in dataset
     for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
          # Convert img to tensor
          img = torch.from_numpy(img).to(device=device)
          img = img.half() if half else img.float()
          img /= 255.0

          # Check if image has 3 dimensions. If not, add an extra dimension
          if img.ndimension()==3:
               # Simulate a batch dimension also
               # Earlier (3, 224, 224) => (1, 3, 224, 224)          
               img = img.unsqueeze(dim=0)
          
          t1 = time_synchronized()

          # Perform inference on the model using processed image
          pred = model(img, augment)[0]

          # Apply NMS
          pred = non_max_suppression(
               prediction=pred,
               conf_thres=conf_thres,
               iou_thres=iou_thres,
               classes=classes,
               agnostic=agnostic_nms,
               max_det=max_det
          )

          # Record time after NMS
          t2 = time_synchronized()

          if classify:
               pred = apply_classifier(x=pred, model=modelc, img=img, im0=im0s)

     # Process detection
     # iterate through detections per image
     for i, det in enumerate(pred):
          if webcam:
               pass










