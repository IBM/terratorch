"""EuroSAT dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.utils import Path, check_integrity, download_url, extract_archive, rasterio_loader

from torchgeo.datasets.eurosat import EuroSAT as EuroSAT_

class EuroSAT(EuroSAT_):
    """EuroSAT dataset.

    The `EuroSAT <https://github.com/phelber/EuroSAT>`__ dataset is based on Sentinel-2
    satellite images covering 13 spectral bands and consists of 10 target classes with
    a total of 27,000 labeled and geo-referenced images.

    Dataset format:

    * rasters are 13-channel GeoTiffs
    * labels are values in the range [0,9]

    Dataset classes:

    * Annual Crop
    * Forest
    * Herbaceous Vegetation
    * Highway
    * Industrial Buildings
    * Pasture
    * Permanent Crop
    * Residential Buildings
    * River
    * Sea & Lake

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following papers:

    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248
    """

    url = 'https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/'
    filename = 'EuroSATallBands.zip'
    md5 = '5ac12b3b2557aa56e1826e981e8e200e'

    # For some reason the class directories are actually nested in this directory
    base_dir = os.path.join(
        'ds', 'images', 'remote_sensing', 'otherDatasets', 'sentinel_2', 'tif'
    )

    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'train': 'eurosat-train.txt',
        'val': 'eurosat-val.txt',
        'test': 'eurosat-test.txt',
    }
    split_md5s: ClassVar[dict[str, str]] = {
        'train': '908f142e73d6acdf3f482c5e80d851b1',
        'val': '95de90f2aa998f70a3b2416bfe0687b4',
        'test': '7ae5ab94471417b6e315763121e67c5f',
    }

    all_band_names = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B09',
        'B10',
        'B11',
        'B12',
        'B8A',
    )

    rgb_bands = ('B04', 'B03', 'B02')

    BAND_SETS: ClassVar[dict[str, tuple[str, ...]]] = {
        'all': all_band_names,
        'rgb': rgb_bands,
    }


    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = BAND_SETS['all'],
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:

        super().__init__(
            root=root,
            split=split,
            bands=bands,
            transforms=transforms,
            download=download,
            checksum=checksum)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        #sample = {'image': image, 'mask': mask}
        sample = (image, label)

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample



