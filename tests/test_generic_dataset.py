import os

import pytest
import torch
from _pytest.tmpdir import TempPathFactory

from terratorch.datasets import GenericNonGeoPixelwiseRegressionDataset, GenericNonGeoSegmentationDataset

REGRESSION_IMAGE_PATH = "tests/regression_test_input.tif"
REGRESSION_LABEL_PATH = "tests/regression_test_label.tif"
SEGMENTATION_IMAGE_PATH = "tests/segmentation_test_input.tif"
SEGMENTATION_LABEL_PATH = "tests/segmentation_test_label.tif"
NUM_CLASSES_SEGMENTATION = 2


@pytest.fixture(scope="session")
def split_file_path(tmp_path_factory):
    split_file_path = tmp_path_factory.mktemp("split") / "split.txt"
    with open(split_file_path, "w") as f:
        for i in range(5):
            f.write(f"{i}\n")
    return split_file_path


# test file discovery
class TestGenericRegressionDataset:
    @pytest.fixture(scope="class")
    def data_root_regression(self, tmp_path_factory: TempPathFactory):
        data_dir = tmp_path_factory.mktemp("data")
        image_dir_path = data_dir / "input_data"
        label_dir_path = data_dir / "label_data"
        os.mkdir(image_dir_path)
        os.mkdir(label_dir_path)
        for i in range(10):
            os.symlink(REGRESSION_IMAGE_PATH, image_dir_path / f"{i}_img.tif")
            os.symlink(REGRESSION_LABEL_PATH, label_dir_path / f"{i}_label.tif")

        # add a few with no suffix
        for i in range(10, 15):
            os.symlink(REGRESSION_IMAGE_PATH, image_dir_path / f"{i}.tif")
            os.symlink(REGRESSION_LABEL_PATH, label_dir_path / f"{i}.tif")
        return data_dir

    @pytest.fixture(scope="class")
    def regression_dataset(self, data_root_regression, split_file_path):
        return GenericNonGeoPixelwiseRegressionDataset(
            data_root_regression,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        )

    def test_file_discovery_generic_regression_dataset(self, regression_dataset):
        assert len(regression_dataset) == 5

    def test_data_type_regression_float_float(self, regression_dataset):
        assert torch.is_floating_point(regression_dataset[0]["image"])
        assert torch.is_floating_point(regression_dataset[0]["mask"])


class TestGenericSegmentationDataset:
    @pytest.fixture(scope="class")
    def data_root_segmentation(self, tmp_path_factory: TempPathFactory):
        data_dir = tmp_path_factory.mktemp("data")
        image_dir_path = data_dir / "input_data"
        label_dir_path = data_dir / "label_data"
        os.mkdir(image_dir_path)
        os.mkdir(label_dir_path)
        for i in range(10):
            os.symlink(SEGMENTATION_IMAGE_PATH, image_dir_path / f"{i}_img.tif")
            os.symlink(SEGMENTATION_LABEL_PATH, label_dir_path / f"{i}_label.tif")

        # add a few with no suffix
        for i in range(10, 15):
            os.symlink(SEGMENTATION_IMAGE_PATH, image_dir_path / f"{i}.tif")
            os.symlink(SEGMENTATION_LABEL_PATH, label_dir_path / f"{i}.tif")
        return data_dir

    @pytest.fixture(scope="class")
    def segmentation_dataset(self, data_root_segmentation, split_file_path):
        return GenericNonGeoSegmentationDataset(
            data_root_segmentation,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            num_classes=NUM_CLASSES_SEGMENTATION,
            split=split_file_path,
        )

    def test_file_discovery_generic_segmentation_dataset(self, segmentation_dataset):
        assert len(segmentation_dataset) == 5

    def test_data_type_regression_float_long(self, segmentation_dataset):
        assert torch.is_floating_point(segmentation_dataset[0]["image"])
        assert not torch.is_floating_point(segmentation_dataset[0]["mask"])
