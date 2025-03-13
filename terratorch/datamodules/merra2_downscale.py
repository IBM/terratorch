from torch._tensor import Tensor
from granitewxc.datasets.merra2 import Merra2DownscaleDataset
from torchgeo.datamodules import NonGeoDataModule
from typing import Any, Callable, Optional
from granitewxc.datasets.merra2 import Merra2DownscaleDataset
from granitewxc.utils.config import ExperimentConfig
from torch.utils.data.dataloader import DataLoader
from torch._tensor import Tensor
from typing import Callable

class Merra2DownscaleNonGeoDataModule(NonGeoDataModule):

    def __init__(self,
                 data_path_surface: str,
                 data_path_vertical: str,
                 output_vars: list[str],
                 input_surface_vars: Optional[list[str]] = None,
                 input_static_surface_vars: Optional[list[str]] = None,
                 input_vertical_vars: Optional[list[str]] = None,
                 input_levels: Optional[list[float]] = None,
                 time_range: slice = None,
                 climatology_path_surface: Optional[str] = None,
                 climatology_path_vertical: Optional[str] = None,
                 transforms: list[Callable] = [],
                 n_input_timestamps = 1,
                 **kwargs: Any) -> None:
        
          super().__init__(Merra2DownscaleDataset,
                           time_range=time_range,
                           data_path_surface=data_path_surface,
                           data_path_vertical=data_path_vertical,
                           climatology_path_surface=climatology_path_surface,
                           climatology_path_vertical=climatology_path_vertical,
                           input_surface_vars=input_surface_vars,
                           input_static_surface_vars=input_static_surface_vars,
                           input_vertical_vars=input_vertical_vars,
                           input_levels=input_levels,
                           n_input_timestamps=n_input_timestamps,
                           output_vars=output_vars,
                           transforms=transforms,
                           **kwargs)

          self.aug = lambda x: x

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        return super()._dataloader_factory(split)
    
    def setup(self, stage: str) -> None:

        if (stage == 'train'):
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            ) 
        if (stage == 'val'):
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            ) 
        if (stage == 'test'):
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            ) 
        if (stage == 'predict'):
            self.predict_dataset = self.dataset_class(  # type: ignore[call-arg]
                **self.kwargs
            ) 

        return super().setup(stage)
