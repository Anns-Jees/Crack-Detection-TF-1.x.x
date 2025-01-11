"""
Mask R-CNN
Train on the crack dataset and implement crack detection.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Revised by M. Rakeh Saleem

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 crack.py train --dataset=/path/to/crack/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 crack.py train --dataset=/path/to/crack/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 crack.py train --dataset=/path/to/crack/dataset --weights=imagenet

    # Apply color splash to an image
    python3 crack.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 crack.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CrackConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "crack"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 27  # Background + crack

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75

    
#config = CrackConfig()
#dataset_val = CrackDataset()

model_inference = modellib.MaskRCNN(mode="inference", config=CrackConfig(), model_dir=DEFAULT_LOGS_DIR)
#model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)


############################################################
#  Dataset
############################################################
class CrackDataset(utils.Dataset):

    def load_crack(self, dataset_dir, subset):
        """Load a subset of the crack dataset using bounding boxes and dynamically add classes."""
        
        # Load annotations (we expect a list of annotations in COCO format)
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        # Dynamically add classes based on the 'categories' field in the annotations
        categories = annotations['categories']  # Get the list of categories
        for category in categories:
            class_id = category['id']
            class_name = category['name']
            self.add_class("crack", class_id, class_name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load images and bounding boxes
        images = annotations['images']  # Images metadata
        bboxes = annotations['annotations']  # Bounding boxes

        # Add images and corresponding bounding boxes
        for image_info in images:
            image_id = image_info['id']
            image_path = os.path.join(dataset_dir, image_info['file_name'])
            height = image_info['height']
            width = image_info['width']

            # Get the bounding boxes for this image
            image_bboxes = [bbox for bbox in bboxes if bbox['image_id'] == image_id]

            # Add the image
            self.add_image(
                "crack",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                bboxes=image_bboxes  # Attach bounding boxes
            )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a crack dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "crack":
            return super(self.__class__, self).load_mask(image_id)

        # **Changes start here**: Use bounding boxes to create masks.
        bboxes = image_info['bboxes']  # Get bounding boxes from image_info
        mask = np.zeros([image_info["height"], image_info["width"], len(bboxes)], dtype=np.uint8)

        # Loop through each bounding box and create a rectangular mask
        for i, bbox in enumerate(bboxes):
            y1, x1, y2, x2 = bbox['bbox']  # Assuming the format is [ymin, xmin, ymax, xmax]
            mask[y1:y2, x1:x2, i] = 1  # Set pixels inside the bounding box to 1

        # **Changes end here**: Return the mask and class IDs.
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "crack":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CrackDataset()
    dataset_train.load_crack(args.dataset, "train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CrackDataset()
    dataset_val.load_crack(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    
    #mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_val, 1,
    #                                                                    verbose=1)    
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers='heads')
                #custom_callbacks=[mean_average_precision_callback])
    
  #  model.train(dataset_train, dataset_val, 
  #              learning_rate=config.LEARNING_RATE / 10, 
  #              epochs=100, 
  #              layers='heads')
  #  model.train(dataset_train, dataset_val, 
  #              learning_rate=config.LEARNING_RATE,
  #              epochs=200, 

  #              layers='all')
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse


    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cracks.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/crack/dataset/",
                        help='Directory of the Crack dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CrackConfig()
    else:
        class InferenceConfig(CrackConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
      # Exclude layers that are dependent on the number of classes
      model.load_weights(weights_path, 
                        by_name=True, 
                        exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask", "mrcnn_class"])


    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
