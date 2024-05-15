import numpy as np
from collections.abc import Sequence
import torch
from pathlib import Path
import json
import h5py

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchgeo.datasets import NonGeoDataset
from terratorch.datasets.utils import to_tensor

import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize


class MSACropTypeNonGeo(NonGeoDataset):

    all_band_names = (
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
        "CLOUD_PROBABILITY"
    )

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(self, data_root: str, bands: Sequence[str] = BAND_SETS["all"], transform: A.Compose | None = None, split="train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)
        if split == "val":
            split = "valid"
        
        self.transform = transform if transform else lambda **batch: to_tensor(batch)
        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = np.array(
            [self.all_band_names.index(b) for b in bands if b in self.all_band_names]
        )
        self.split = split
        data_root = Path(data_root)
        self.data_directory = data_root / "m-SA-crop-type"

        partition_file = self.data_directory / "default_partition.json"
        with open(partition_file, 'r') as file:
            partitions = json.load(file)

        if split not in partitions:
            raise ValueError(f"Split '{split}' not found.")

        self.image_files = [self.data_directory / (filename + ".hdf5") for filename in partitions[split]]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path = self.image_files[index]
        
        with h5py.File(file_path, 'r') as h5file:
            keys = sorted(h5file.keys())
            keys = np.array([key for key in keys if key != 'label'])[self.band_indices]
            bands = [np.array(h5file[key]) for key in keys]

            image = np.stack(bands, axis=-1)
            mask = np.array(h5file['label'])    
        
        output =  {
            "image": image.astype(np.float32),
            "mask": mask
        }

        output = self.transform(**output)
        output['mask'] = output['mask'].long()
            
        return output

    def _validate_bands(self, bands: Sequence[str]) -> None:
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def __len__(self):
        return len(self.image_files)
        
    def plot(self, arg, suptitle: str | None = None) -> None:
        if isinstance(arg, int):
            sample = self.__getitem__(arg)
        elif isinstance(arg, dict):
            sample = arg
        else:
            raise TypeError("Argument must be an integer index or a sample dictionary.")
            
        showing_predictions = sample['prediction'] if "prediction" in sample else None
        
        self.plot_sample(sample, showing_predictions, suptitle)

    def plot_sample(self, sample, prediction=None, suptitle: str | None = None, class_names=None):
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")
        
        image = sample["image"].numpy()  
        image = image[rgb_indices, :, :]
        image = np.transpose(image, (1, 2, 0))  
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)
        
        mask = sample["mask"].numpy()
        num_classes = len(np.unique(mask))  
        
        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(num_images * 4, 4), tight_layout=True)
        
        cmap = colormaps["jet"]
        norm = Normalize(vmin=0, vmax=num_classes - 1)
        
        ax[0].imshow(image)  
        ax[0].set_title("Image")
        ax[0].axis("off")
        
        ax[1].imshow(mask, cmap=cmap, norm=norm)
        ax[1].set_title("Ground Truth Mask")
        ax[1].axis("off")
        
        ax[2].imshow(image)  
        ax[2].imshow(mask, cmap=cmap, alpha=0.3, norm=norm) 
        ax[2].set_title("GT Mask on Image")
        ax[2].axis("off")
        
        if prediction is not None:
            prediction = prediction.numpy()
            ax[3].imshow(prediction, cmap=cmap, norm=norm)
            ax[3].set_title("Predicted Mask")
            ax[3].axis("off")
        
        if class_names:
            legend_handles = [mpatches.Patch(color=cmap(i), label=class_names[i]) for i in range(num_classes)]
            ax[0].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if suptitle:
            plt.suptitle(suptitle)
        
        return fig
        