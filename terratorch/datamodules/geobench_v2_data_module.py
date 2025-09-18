"""Ugly proxy objects so parsing config file works with transforms.

    These are necessary since, for LightningCLI to instantiate arguments as
    objects from the config, they must have type annotations

    In TorchGeo, `transforms` is passed in **kwargs, so it has no type annotations!
    To get around that, we create these wrappers that have transforms type annotated.
    They create the transforms and forward all method and attribute calls to the
    original TorchGeo datamodule.

    Additionally, TorchGeo datasets pass the data to the transforms callable
    as a dict, and as a tensor.

    Albumentations expects this data not as a dict but as different key-value
    arguments, and as numpy. We handle that conversion here. 
"""
from collections.abc import Callable
from typing import Any, Union
from functools import partial
import numpy as np
from torch import Tensor
from torch.utils.data import default_collate
import kornia.augmentation as K
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
import torch.nn as nn
from terratorch.datasets.transforms import kornia_augmentations_to_callable_with_dict
try:
    from geobench_v2.datamodules import (
        GeoBenchClassificationDataModule, 
        GeoBenchSegmentationDataModule, 
        GeoBenchObjectDetectionDataModule,
        MultiTemporalSegmentationAugmentation,
        MultiModalSegmentationAugmentation,
        )
except ImportError as e:
    import logging
    logging.getLogger("terratorch").debug("geobench_v2 not installed")


class GeoBenchV2ClassificationDataModule(GeoBenchClassificationDataModule):
    """Proxy object for using Classification DataModules defined by geobench_v2.

    Allows for transforms to be defined and passed using config files.
    The only reason this class exists is so that we can annotate the transforms argument with a type.
    This is required for lightningcli and config files.
    As such, all getattr and setattr will be redirected to the underlying class.
    """

    def __init__(
        self,
        cls: type[GeoBenchClassificationDataModule],
        img_size: int,
        band_order: list | dict,
        batch_size: int | None = None,
        num_workers: int = 0,
        train_augmentations: None | list[Union[GeometricAugmentationBase2D, K.VideoSequential, IntensityAugmentationBase2D]] | str = "default",
        eval_augmentations: None | list[Union[GeometricAugmentationBase2D, K.VideoSequential, IntensityAugmentationBase2D]] | str = "default",
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoBenchClassificationDataModule]): geobench_v2 Classification DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[Union[GeometricAugmentationBase2D, K.VideoSequential]], optional): List of Albumentations Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if isinstance(train_augmentations, str):
            msg = "If train_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            assert train_augmentations in ["default", "multi_temporal_default"], msg
            
        if isinstance(eval_augmentations, str):
            msg = "If eval_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            assert eval_augmentations in ["default", "multi_temporal_default"], msg

        kwargs["img_size"] = img_size
        kwargs["band_order"] = band_order

        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if not train_augmentations in [None, "default", "multi_temporal_default"]:
            train_augmentations = kornia_augmentations_to_callable_with_dict(train_augmentations)
        kwargs["train_augmentations"] = train_augmentations
            
        if not eval_augmentations in [None, "default", "multi_temporal_default"]:
            eval_augmentations = kornia_augmentations_to_callable_with_dict(eval_augmentations)
        kwargs["eval_augmentations"] = eval_augmentations
        
        if isinstance(band_order, list): 
            if len(band_order) > 0:
                if isinstance(band_order[0], dict):
                    band_order_dict = {}
                    for modality in band_order:
                        band_order_dict.update(modality)
                    band_order = band_order_dict
                    kwargs["band_order"] = band_order

        self._proxy = cls(num_workers=num_workers, **kwargs)
        super().__init__(
            dataset_class = self._proxy.dataset_class, 
            img_size = img_size,
            band_order = band_order)

    @property
    def collate_fn(self):
        return self._proxy.collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._proxy.collate_fn = value

    def setup(self, stage: str):
        return self._proxy.setup(stage)

    def train_dataloader(self):
        return self._proxy.train_dataloader()

    def val_dataloader(self):
        return self._proxy.val_dataloader()

    def test_dataloader(self):
        return self._proxy.test_dataloader()

    def _valid_attribute(self, *args: str):
        return self._proxy._valid_attribute(args)

    def plot(self, args):
        return self._proxy.visualize_batch(args)



class GeoBenchV2ObjectDetectionDataModule(GeoBenchObjectDetectionDataModule):
    """Proxy object for using Object Detection DataModules defined by geobench_v2.

    Allows for transforms to be defined and passed using config files.
    The only reason this class exists is so that we can annotate the transforms argument with a type.
    This is required for lightningcli and config files.
    As such, all getattr and setattr will be redirected to the underlying class.
    """

    def __init__(
        self,
        cls: type[GeoBenchObjectDetectionDataModule],
        img_size: int,
        band_order: list,
        batch_size: int | None = None,
        num_workers: int = 0,
        train_augmentations: None | list[Union[GeometricAugmentationBase2D, K.VideoSequential, IntensityAugmentationBase2D]]| str = "default",
        eval_augmentations: None | list[Union[GeometricAugmentationBase2D, K.VideoSequential, IntensityAugmentationBase2D]] | str = "default",
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoBenchObjectDetectionDataModule]): geobench_v2 Object Detection DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[Union[GeometricAugmentationBase2D, K.VideoSequential]], optional): List of Albumentations Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if isinstance(train_augmentations, str):
            msg = "If train_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            assert train_augmentations in ["default", "multi_temporal_default"], msg
            
        if isinstance(eval_augmentations, str):
            msg = "If eval_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            assert eval_augmentations in ["default", "multi_temporal_default"], msg

        kwargs["img_size"] = img_size
        kwargs["band_order"] = band_order

        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if not train_augmentations in [None, "default", "multi_temporal_default"]:
            train_augmentations = kornia_augmentations_to_callable_with_dict(train_augmentations)
        kwargs["train_augmentations"] = train_augmentations
            
        if not eval_augmentations in [None, "default", "multi_temporal_default"]:
            eval_augmentations = kornia_augmentations_to_callable_with_dict(eval_augmentations)
        kwargs["eval_augmentations"] = eval_augmentations
        
        if len(band_order) > 0:
            if isinstance(band_order[0], dict):
                band_order_dict = {}
                for modality in band_order:
                    band_order_dict.update(modality)
                band_order = band_order_dict
                kwargs["band_order"] = band_order

        self._proxy = cls(num_workers=num_workers, **kwargs)
        super().__init__(
            dataset_class = self._proxy.dataset_class, 
            img_size = img_size,
            band_order = band_order)  # dummy arg

    @property
    def collate_fn(self):
        return self._proxy.collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._proxy.collate_fn = value

    @property
    def patch_size(self):
        return self._proxy.patch_size

    @property
    def length(self):
        return self._proxy.length

    def setup(self, stage: str):
        return self._proxy.setup(stage)

    def train_dataloader(self):
        return self._proxy.train_dataloader()

    def val_dataloader(self):
        return self._proxy.val_dataloader()

    def test_dataloader(self):
        return self._proxy.test_dataloader()

    def _valid_attribute(self, *args: str):
        return self._proxy._valid_attribute(args)



class GeoBenchV2SegmentationDataModule(GeoBenchSegmentationDataModule):
    """Proxy object for using Segmentation DataModules defined by geobench_v2.

    Allows for transforms to be defined and passed using config files.
    The only reason this class exists is so that we can annotate the transforms argument with a type.
    This is required for lightningcli and config files.
    As such, all getattr and setattr will be redirected to the underlying class.
    """

    def __init__(
        self,
        cls: type[GeoBenchSegmentationDataModule],
        img_size: int,
        band_order: list,
        batch_size: int | None = None,
        num_workers: int = 0,
        train_augmentations: None | list[Union[GeometricAugmentationBase2D, K.VideoSequential, IntensityAugmentationBase2D]] | str = "default",
        eval_augmentations: None | list[Union[GeometricAugmentationBase2D, K.VideoSequential, IntensityAugmentationBase2D]] | str = "default",
        collate_fn: Callable = None,
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoDataModule]): geobench_v2 Segmentation DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[Union[GeometricAugmentationBase2D, K.VideoSequential]], optional): List of Kornia Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if isinstance(train_augmentations, str):
            msg = "If train_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            assert train_augmentations in ["default", "multi_temporal_default"], msg
            
        if isinstance(eval_augmentations, str):
            msg = "If eval_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            assert eval_augmentations in ["default", "multi_temporal_default"], msg
            
        kwargs["img_size"] = img_size
        kwargs["band_order"] = band_order
        
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if "rename_modalities" in kwargs:
            if train_augmentations == "default":
                train_augmentations = [ K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5)]
            elif train_augmentations == "multi_temporal_default":
                train_augmentations = [K.VideoSequential(), K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5)]
            elif train_augmentations is None:
                train_augmentations = [nn.Identity()]
            train_augmentations = kornia_augmentations_to_callable_with_dict(train_augmentations)
            train_augmentations = MultiModalSegmentationAugmentation(transforms=train_augmentations)

            if eval_augmentations in ["default", None]:
                eval_augmentations = [nn.Identity()]
            elif eval_augmentations == "multi_temporal_default":
                eval_augmentations = [K.VideoSequential(), nn.Identity()]
            eval_augmentations = kornia_augmentations_to_callable_with_dict(eval_augmentations)
            eval_augmentations = MultiModalSegmentationAugmentation(transforms=eval_augmentations)
        else:
            if not train_augmentations in [None, "default", "multi_temporal_default"]:
                if isinstance(train_augmentations[0], K.VideoSequential):
                    train_augmentations = kornia_augmentations_to_callable_with_dict(train_augmentations)
                    train_augmentations = MultiTemporalSegmentationAugmentation(transforms=train_augmentations)
                else:
                    train_augmentations = kornia_augmentations_to_callable_with_dict(train_augmentations)
            
            if not eval_augmentations in [None, "default", "multi_temporal_default"]:
                if isinstance(eval_augmentations[0], K.VideoSequential):
                    eval_augmentations = kornia_augmentations_to_callable_with_dict(eval_augmentations)
                    eval_augmentations = MultiTemporalSegmentationAugmentation(transforms=eval_augmentations)
                else:
                    eval_augmentations = kornia_augmentations_to_callable_with_dict(eval_augmentations)

        kwargs["train_augmentations"] = train_augmentations
        kwargs["eval_augmentations"] = eval_augmentations

        if len(band_order) > 0:
            if isinstance(band_order[0], dict):
                band_order_dict = {}
                for modality in band_order:
                    band_order_dict.update(modality)
                band_order = band_order_dict
                kwargs["band_order"] = band_order

        self._proxy = cls(num_workers=num_workers, **kwargs)

        super().__init__(
            dataset_class = self._proxy.dataset_class, 
            img_size = img_size,
            band_order = band_order
            )  # dummy arg


        self.collate_fn = collate_fn

    @property
    def collate_fn(self):
        return self._proxy.collate_fn

    @collate_fn.setter
    def collate_fn(self, value):
        self._proxy.collate_fn = value

    @property
    def patch_size(self):
        return self._proxy.patch_size

    @property
    def length(self):
        return self._proxy.length

    @property
    def val_dataset(self):
        return self._proxy.val_dataset

    def setup(self, stage: str):
        return self._proxy.setup(stage)

    def train_dataloader(self):
        return self._proxy.train_dataloader()

    def val_dataloader(self):
        return self._proxy.val_dataloader()

    def test_dataloader(self):
        return self._proxy.test_dataloader()

    def _valid_attribute(self, *args: str):
        return self._proxy._valid_attribute(args)

    def plot(self, args):
        return self._proxy.visualize_batch(args)







            