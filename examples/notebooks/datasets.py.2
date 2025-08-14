# Copyright (c) 2020 Akihiro Nitta
# All rights reserved.

from __future__ import print_function, division
import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from collections import defaultdict

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
    
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(XviewDataset, self).__init__(root, transforms, transform, target_transform)
        image_ids = [int(f.replace(".tif", "")) for f in os.listdir(root)]

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Go to <http://xviewdataset.org/> to download it.')
        
        if not os.path.isfile(annFile):
            raise RuntimeError('Annotation file not found or corrupted.' +
                               ' See <http://xviewdataset.org/> to download it.')
        
        # === load annotations ===
        import json
        import time
        with open(annFile, "r") as f:
            print("loading annotations into memory...")
            t0 = time.time()
            raw_anns = json.load(f)
            raw_anns = raw_anns["features"]
            t1 = time.time()
            print(f"Done (t={t1-t0:.2f}s)")
            
        # === create index ===
        print("creating index...")
        image_id_to_object_ids = defaultdict(list)
        objects = []
        for i in range(len(raw_anns)): # for all objects
            image_id = int(raw_anns[i]["properties"]["image_id"].replace(".tif", ""))
            xmin, ymin, xmax, ymax = map(int, raw_anns[i]["properties"]["bounds_imcoords"].split(","))
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            type_id = raw_anns[i]["properties"]["type_id"]
            image_id_to_object_ids[image_id].append(i)
            objects.append({
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "category_id": type_id
            })
        print("index created!")
        self.image_ids = image_ids
        self.image_id_to_object_ids = image_id_to_object_ids
        self.objects = objects
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is a dictionary.
        """
        img_id = self.image_ids[index]
        ann_ids = self.image_id_to_object_ids[img_id]
        target = []
        for ann_id in ann_ids:
            target.append(self.objects[ann_id])

        fname = os.path.join(self.root, str(img_id) + ".tif")
        
        img = Image.open(os.path.join(self.root, fname)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    ds = XviewDataset(root="/home/nitta/data/xview/train_images",
                      annFile="/home/nitta/data/xview/xView_train.geojson")
    img, target = ds[0]
    print(img)
    print(target)
