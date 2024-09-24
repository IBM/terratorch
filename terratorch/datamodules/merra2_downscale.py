from torch._tensor import Tensor
from granitewxc.datasets.merra2 import Merra2DownscaleDataset
from torchgeo.datamodules import NonGeoDataModule
from typing import Any
from granitewxc.datasets.merra2 import Merra2DownscaleDataset
from torch.utils.data.dataloader import DataLoader
from torch._tensor import Tensor

class Merra2DownscaleDatasetTerraTorch(Merra2DownscaleDataset):
    def __init__(self, split : str, **kwargs):
        super().__init__(**kwargs)
        self.split = split

    def __getitem__(self, index) -> dict[Tensor | int]:
        the_way_it_is = super().__getitem__(index)
        the_way_it_should_be = {}
        the_way_it_should_be['image'] = the_way_it_is
        the_way_it_should_be['mask'] = the_way_it_is
        the_way_it_should_be['filename'] = f'{index}.tif'

        return the_way_it_should_be

class Merra2DownscaleNonGeoDataModule(NonGeoDataModule):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(Merra2DownscaleDatasetTerraTorch, **kwargs)
        self.aug = lambda x: x

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        return super()._dataloader_factory(split)
    
    def setup(self, stage: str) -> None:
        if (stage == 'predict'):
            self.predict_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='predict', **self.kwargs
            ) 
        return super().setup(stage)