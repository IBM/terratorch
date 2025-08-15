# Copyright (c) 2020 Akihiro Nitta
# All rights reserved.

from __future__ import print_function, division
import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from collections import defaultdict
import logging
import json
import collections
import torch



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

LABEL_TO_STRING = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Passenger/Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck Tractor w/ Box Trailer",
    26: "Truck Tractor",
    27: "Trailer",
    28: "Truck Tractor w/ Flatbed Trailer",
    29: "Truck Tractor w/ Liquid Tank",
    32: "Crane Truck",
    33: "Railway Vehicle",
    34: "Passenger Car",
    35: "Cargo/Container Car",
    36: "Flat Car",
    37: "Tank car",
    38: "Locomotive",
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
    53: "Engineering Vehicle",
    54: "Tower crane",
    55: "Container Crane",
    56: "Reach Stacker",
    57: "Straddle Carrier",
    59: "Mobile Crane",
    60: "Dump Truck",
    61: "Haul Truck",
    62: "Scraper/Tractor",
    63: "Front loader/Bulldozer",
    64: "Excavator",
    65: "Cement Mixer",
    66: "Ground Grader",
    71: "Hut/Tent",
    72: "Shed",
    73: "Building",
    74: "Aircraft Hangar",
    76: "Damaged Building",
    77: "Facility",
    79: "Construction Site",
    83: "Vehicle Lot",
    84: "Helipad",
    86: "Storage Tank",
    89: "Shipping container lot",
    91: "Shipping Container",
    93: "Pylon",
    94: "Tower",
}

class XviewDataset(VisionDataset):
    """xView object detection dataset.
    
    Args:
        root_dir (string): Directory with all the images.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    
    def __init__(self, root, annFile, transform=None):
        import os, json, collections

        self.root = root
        self.annFile = annFile
        self.transform = transform

        # Load annotations
        with open(self.annFile) as f:
            anns = json.load(f)

        self.objects = {}  # object_id -> object
        self.ids = set()

        for idx, feat in enumerate(anns['features']):
            obj_id = idx
            image_id = feat['properties']['image_id']

            img_path = os.path.join(self.root, f"{image_id}")
            if not os.path.exists(img_path):
                logging.info(f'{img_path} skipped because it was not found')
                continue

            self.ids.add(image_id)
            self.objects[obj_id] = {
                "image_id": image_id,
                "bbox": feat['properties']['bounds_imcoords'],
                "type_id": feat['properties']['type_id']
            }

        # Create mapping from image_id to all object_ids
        self.image_id_to_object_ids = collections.defaultdict(list)
        for obj_id, obj in self.objects.items():
            self.image_id_to_object_ids[obj["image_id"]].append(obj_id)

        # Keep ids only for images that still exist
        self.ids = list(self.ids)


    def filter_invalid_boxes(self, target):
        boxes = target["boxes"]
        # keep only boxes with strictly positive width & height
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        target["boxes"] = boxes[keep]
        if "labels" in target:
            target["labels"] = target["labels"][keep]
        return target

    def target_list_to_dict(self, target_list):
        if len(target_list) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        boxes = torch.tensor([list(map(float, t['bbox'].split(','))) for t in target_list], dtype=torch.float32)
        labels = torch.tensor([t['type_id'] for t in target_list], dtype=torch.int64)
        return {"boxes": boxes, "labels": labels}


    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.image_id_to_object_ids[img_id]
        target = [self.objects[ann_id] for ann_id in ann_ids]

        target = self.target_list_to_dict(target)
        target = self.filter_invalid_boxes(target)

        fname = os.path.join(self.root, img_id)  # don't append .tif if it's already in img_id
        img = Image.open(fname).convert("RGB")

        if self.transform is not None :
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

