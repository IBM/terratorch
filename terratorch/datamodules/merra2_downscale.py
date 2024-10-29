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
        batch = super().__getitem__(index)
        batch_extended = {}
        batch_extended['mask'] = batch.pop("y")
        batch_extended['image'] = batch
        batch_extended['filename'] = f'{index}.tif'
      
        return batch_extended

class Merra2DownscaleNonGeoDataModule(NonGeoDataModule):

    def __init__(self, input_surface_vars: list[int | tuple[int, int] | str] | None = None,
                       input_static_surface_vars: list[int | tuple[int, int] | str] | None = None,
                       input_vertical_vars: list[int | tuple[int, int] | str] | None = None,
                       input_levels: list[float] = None,
                       output_vars: list[int | tuple[int, int] | str] | None = None,
                       n_input_timestamps: int = 2,
                       crop_lat: list[float] = None,
                       input_size_lat: int = 60,
                       input_size_lon: int = 60,
                       apply_smoothen: bool = True, 
                       **kwargs: Any) -> None:
        
        super().__init__(Merra2DownscaleDatasetTerraTorch,
                         input_surface_vars=input_surface_vars,
                         input_static_surface_vars=input_static_surface_vars,
                         input_vertical_vars=input_vertical_vars,
                         output_vars=output_vars,
                         n_input_timestamps=n_input_timestamps,
                         crop_lat=crop_lat,
                         input_size_lat=input_size_lat,
                         input_size_lon=input_size_lon,
                         apply_smoothen=apply_smoothen,
                         **kwargs)

        self.aug = lambda x: x

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        return super()._dataloader_factory(split)
    
    def setup(self, stage: str) -> None:

        if (stage == 'train'):
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='train', **self.kwargs
            ) 
        if (stage == 'val'):
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='val', **self.kwargs
            ) 
        if (stage == 'test'):
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='test', **self.kwargs
            ) 
        if (stage == 'predict'):
            self.predict_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='predict', **self.kwargs
            ) 

        return super().setup(stage)
