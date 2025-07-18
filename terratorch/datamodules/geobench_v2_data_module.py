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
from typing import Any
from functools import partial


import pdb
import numpy as np
from torch import Tensor
from torch.utils.data import default_collate
import kornia.augmentation as K
from kornia.augmentation._2d.geometric.base import GeometricAugmentationBase2D


from geobench_v2.datamodules import GeoBenchClassificationDataModule, GeoBenchSegmentationDataModule, GeoBenchObjectDetectionDataModule

from terratorch.datasets.transforms import kornia_augmentations_to_callable_with_dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torchgeo.datasets.utils import percentile_normalization, lazy_import
from matplotlib import patches


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
        train_augmentations: None | list[GeometricAugmentationBase2D, K.VideoSequential] | str = "default",
        eval_augmentations: None | list[GeometricAugmentationBase2D, K.VideoSequential] | str = "default",
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoBenchClassificationDataModule]): geobench_v2 Classification DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[GeometricAugmentationBase2D, K.VideoSequential], optional): List of Albumentations Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if isinstance(train_augmentations, str):
            assert train_augmentations in ["default", "multi_temporal_default"],"If train_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
        if isinstance(eval_augmentations, str):
            assert eval_augmentations in ["default", "multi_temporal_default"],"If eval_augmentations is a string, it must be 'default' or 'multi_temporal_default'"

        kwargs["img_size"] = img_size
        kwargs["band_order"] = band_order

        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if not train_augmentations in [None, "default", "multi_temporal_default"]:
            kwargs["train_augmentations"] =kornia_augmentations_to_callable_with_dict(train_augmentations)
        else:
            kwargs["train_augmentations"] = train_augmentations
            
        if not eval_augmentations in [None, "default", "multi_temporal_default"]:
            kwargs["eval_augmentations"] =kornia_augmentations_to_callable_with_dict(eval_augmentations)
        else:
            kwargs["eval_augmentations"] = eval_augmentations
        
        if isinstance(band_order, list): #REMOVE THIS FOR FINAL VERSION, IF LIST ASSUME IT CONTAIN BANDS DIRECTLY
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
        root: str,
        img_size: int,
        band_order: Any,
        categories: list,
        batch_size: int | None = None,
        eval_batch_size: int | None = None,
        num_workers: int = 0,
        train_augmentations: None | list[GeometricAugmentationBase2D, K.VideoSequential] | str = "default",
        eval_augmentations: None | list[GeometricAugmentationBase2D, K.VideoSequential] | str = "default",
        plot_indexes: list = [0,1,2],
        collate_fn: Callable = None,
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoBenchObjectDetectionDataModule]): geobench_v2 Object Detection DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[GeometricAugmentationBase2D, K.VideoSequential], optional): List of Albumentations Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if isinstance(train_augmentations, str):
            assert train_augmentations in ["default", "multi_temporal_default"],"If train_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
        if isinstance(eval_augmentations, str):
            assert eval_augmentations in ["default", "multi_temporal_default"],"If eval_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
        
        kwargs['root'] = root
        kwargs["img_size"] = img_size
        kwargs["band_order"] = band_order

        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if eval_batch_size is not None:
            kwargs["eval_batch_size"] = batch_size
        if not train_augmentations in [None, "default", "multi_temporal_default"]:
            kwargs["train_augmentations"] =kornia_augmentations_to_callable_with_dict(train_augmentations)
        else:
            kwargs["train_augmentations"] = train_augmentations
            
        if not eval_augmentations in [None, "default", "multi_temporal_default"]:
            kwargs["eval_augmentations"] =kornia_augmentations_to_callable_with_dict(eval_augmentations)
        else:
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
            band_order = band_order,
            batch_size = batch_size,
            eval_batch_size = eval_batch_size)  # dummy arg
        
        self.collate_fn = collate_fn
        self.plot_indexes = plot_indexes
        self.categories = categories

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
    
    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        show_feats: str | None = 'both',
        box_alpha: float = 0.7,
        mask_alpha: float = 0.7,
        confidence_score = 0.5
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            show_feats: optional string to pick features to be shown: boxes, masks, both
            box_alpha: alpha value of box
            mask_alpha: alpha value of mask

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
            DependencyNotFoundError: If plotting masks and scikit-image is not installed.

        .. versionadded:: 0.4
        """
        assert show_feats in {'boxes', 'masks', 'both'}

        image = sample['image']
        image = image.median(1).values if len(image.shape) == 4 else image
        # get indexes to plot 
        image = image[ self.plot_indexes, :, :]
        if image.mean() > 1:
            image = image / 10000
        
        image = percentile_normalization(image.permute(1, 2, 0).numpy())

        if show_feats != 'boxes':
            skimage = lazy_import('skimage')

        boxes = sample['boxes'].cpu().numpy()
        labels = sample['labels'].cpu().numpy()

        if 'masks' in sample:
            masks = [mask.squeeze().cpu().numpy() for mask in sample['masks']]

        n_gt = len(boxes)

        ncols = 1
        show_predictions = 'prediction_labels' in sample

        if show_predictions:
            show_pred_boxes = False
            show_pred_masks = False
            prediction_labels = sample['prediction_labels'].numpy()
            prediction_scores = sample['prediction_scores'].numpy()
            if 'prediction_boxes' in sample:
                prediction_boxes = sample['prediction_boxes'].numpy()
                show_pred_boxes = True
            if 'prediction_masks' in sample:
                prediction_masks = sample['prediction_masks'].numpy()
                show_pred_masks = True

            n_pred = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 13))
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')

        cm = plt.get_cmap('gist_rainbow')
        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / len(self.categories))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            if show_feats in {'boxes', 'both'}:
                r = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle='dashed',
                    edgecolor=color,
                    facecolor='none',
                )
                axs[0, 0].add_patch(r)

            # Add labels
            label = self.categories[class_num]
            caption = label
            axs[0, 0].text(
                x1, y1 - 8, caption, color='white', size=11, backgroundcolor='none'
            )

            # Add masks
            if show_feats in {'masks', 'both'} and 'masks' in sample:
                mask = masks[i]
                contours = skimage.measure.find_contours(mask, 0.5)
                for verts in contours:
                    verts = np.fliplr(verts)
                    p = patches.Polygon(
                        verts, facecolor=color, alpha=mask_alpha, edgecolor='white'
                    )
                    axs[0, 0].add_patch(p)

            if show_titles:
                axs[0, 0].set_title('Ground Truth')

        if show_predictions:
            axs[0, 1].imshow(image)
            axs[0, 1].axis('off')
            for i in range(n_pred):
                score = prediction_scores[i]
                if score < confidence_score:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(self.categories))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle='dashed',
                        edgecolor=color,
                        facecolor='none',
                    )
                    axs[0, 1].add_patch(r)

                    # Add labels
                    label = self.categories[class_num]
                    caption = f'{label} {score:.3f}'
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color='white',
                        size=11,
                        backgroundcolor='none',
                    )

                # Add masks
                if show_pred_masks:

                    mask = prediction_masks[i][0]
                    contours = skimage.measure.find_contours(mask, 0.5)
                    for verts in contours:
                        verts = np.fliplr(verts)
                        p = patches.Polygon(
                            verts, facecolor=color, alpha=mask_alpha, edgecolor='white'
                        )
                        axs[0, 1].add_patch(p)

            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig




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
        train_augmentations: None | list[GeometricAugmentationBase2D, K.VideoSequential] | str = "default",
        eval_augmentations: None | list[GeometricAugmentationBase2D, K.VideoSequential] | str = "default",
        **kwargs: Any,
    ):
        """Constructor

        Args:
            cls (type[GeoDataModule]): geobench_v2 Segmentation DataModule class to be instantiated
            batch_size (int | None, optional): batch_size. Defaults to None.
            num_workers (int, optional): num_workers. Defaults to 0.
            transforms (None | list[GeometricAugmentationBase2D, K.VideoSequential], optional): List of Kornia Transforms.
                Should enc with ToTensorV2. Defaults to None.
            **kwargs (Any): Arguments passed to instantiate `cls`.
        """
        if isinstance(train_augmentations, str):
            assert train_augmentations in ["default", "multi_temporal_default"],"If train_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
        if isinstance(eval_augmentations, str):
            assert eval_augmentations in ["default", "multi_temporal_default"],"If eval_augmentations is a string, it must be 'default' or 'multi_temporal_default'"
            
        kwargs["img_size"] = img_size
        kwargs["band_order"] = band_order
        
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        if not train_augmentations in [None, "default", "multi_temporal_default"]:
            kwargs["train_augmentations"] = kornia_augmentations_to_callable_with_dict(train_augmentations)
        else:
            kwargs["train_augmentations"] = train_augmentations
            
        if not eval_augmentations in [None, "default", "multi_temporal_default"]:
            kwargs["eval_augmentations"] = kornia_augmentations_to_callable_with_dict(eval_augmentations)
        else:
            kwargs["eval_augmentations"] = eval_augmentations

        if len(band_order) > 0:
            print(f"\n\nband_order before: {band_order}")
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







            