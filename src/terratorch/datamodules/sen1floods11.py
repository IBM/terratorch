from pathlib import Path
from typing import Any

import kornia.augmentation as K  # noqa: N812
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential

from terratorch.datasets import Sen1Floods11NonGeo

MEANS = [
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
]

STDS = [
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
]


class Sen1Floods11NonGeoDataModule(NonGeoDataModule):
    """NonGeo Fire Scars data module implementation"""

    def __init__(self, batch_size: int = 4, num_workers: int = 0, **kwargs: Any) -> None:
        super().__init__(Sen1Floods11NonGeo, batch_size, num_workers, **kwargs)
        self.aug = AugmentationSequential(K.Normalize(MEANS, STDS), data_keys=["image", "mask"])
