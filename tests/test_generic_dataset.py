# Copyright contributors to the Terratorch project

import os

import pytest
import torch
from _pytest.tmpdir import TempPathFactory

from terratorch.datasets import GenericNonGeoPixelwiseRegressionDataset, GenericNonGeoSegmentationDataset, HLSBands

REGRESSION_IMAGE_PATH = "tests/resources/inputs/regression_test_input.tif"
REGRESSION_LABEL_PATH = "tests/resources/inputs/regression_test_label.tif"
SEGMENTATION_IMAGE_PATH = "tests/resources/inputs/segmentation_test_input.tif"
SEGMENTATION_LABEL_PATH = "tests/resources/inputs/segmentation_test_label.tif"
NUM_CLASSES_SEGMENTATION = 2

# Testing bands
# HLS_bands
HLS_dataset_bands = [
    "COASTAL_AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
    "CIRRUS",
    "THEMRAL_INFRARED_1",
    "THEMRAL_INFRARED_2",
]

HLS_output_bands = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]

HLS_expected_filter_bands = list(range(1, 7))
# Integer Intervals bands
int_dataset_bands = [(0, 20)]
int_output_bands = [(1, 6), (10, 12)]
# Simple string bands
str_dataset_bands = [f"band_{j}" for j in range(20)]
str_output_bands = [f"band_{j}" for j in range(1, 7)] + [f"band_{j}" for j in range(10, 13)]

expected_filter_indices = list(range(1, 7)) + list(range(10, 13))


# Mixed case
mixed_dataset_bands = [
    (0, 10),
    HLSBands.RED,
    HLSBands.BLUE,
    HLSBands.GREEN,
    "extra_band_1",
    "extra_band_2",
    200,
    201,
    202,
]
mixed_output_bands = [1, 2, HLSBands.BLUE, "extra_band_1", 201, 202]
expected_mixed_filter_indices = [1, 2, 12, 14, 17, 18]


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

    @pytest.fixture(scope="class")
    def regression_dataset_with_HLS_bands(self, data_root_regression, split_file_path):
        return GenericNonGeoPixelwiseRegressionDataset(
            data_root_regression,
            dataset_bands=HLS_dataset_bands,
            output_bands=HLS_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), HLS_expected_filter_bands

    @pytest.fixture(scope="class")
    def regression_dataset_with_interval_bands(self, data_root_regression, split_file_path):
        return GenericNonGeoPixelwiseRegressionDataset(
            data_root_regression,
            dataset_bands=int_dataset_bands,
            output_bands=int_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), expected_filter_indices

    @pytest.fixture(scope="class")
    def regression_dataset_with_str_bands(self, data_root_regression, split_file_path):
        return GenericNonGeoPixelwiseRegressionDataset(
            data_root_regression,
            dataset_bands=str_dataset_bands,
            output_bands=str_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), expected_filter_indices

    @pytest.fixture(scope="class")
    def regression_dataset_with_mixed_bands(self, data_root_regression, split_file_path):
        return GenericNonGeoPixelwiseRegressionDataset(
            data_root_regression,
            dataset_bands=mixed_dataset_bands,
            output_bands=mixed_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), expected_mixed_filter_indices

    @pytest.mark.parametrize(
        "dataset",
        [
            "regression_dataset_with_HLS_bands",
            "regression_dataset_with_str_bands",
            "regression_dataset_with_interval_bands",
            "regression_dataset_with_str_bands",
            "regression_dataset_with_mixed_bands",
        ],
    )
    def test_correct_filter(self, dataset, request):
        fixture, expected = request.getfixturevalue(dataset)
        assert fixture.filter_indices == expected


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

    @pytest.fixture(scope="class")
    def segmentation_dataset_with_HLS_bands(self, data_root_segmentation, split_file_path):
        return GenericNonGeoSegmentationDataset(
            data_root_segmentation,
            NUM_CLASSES_SEGMENTATION,
            dataset_bands=HLS_dataset_bands,
            output_bands=HLS_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), HLS_expected_filter_bands

    @pytest.fixture(scope="class")
    def segmentation_dataset_with_interval_bands(self, data_root_segmentation, split_file_path):
        return GenericNonGeoSegmentationDataset(
            data_root_segmentation,
            NUM_CLASSES_SEGMENTATION,
            dataset_bands=int_dataset_bands,
            output_bands=int_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), expected_filter_indices

    @pytest.fixture(scope="class")
    def segmentation_dataset_with_str_bands(self, data_root_segmentation, split_file_path):
        return GenericNonGeoSegmentationDataset(
            data_root_segmentation,
            NUM_CLASSES_SEGMENTATION,
            dataset_bands=str_dataset_bands,
            output_bands=str_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), expected_filter_indices

    @pytest.fixture(scope="class")
    def segmentation_dataset_with_mixed_bands(self, data_root_segmentation, split_file_path):
        return GenericNonGeoSegmentationDataset(
            data_root_segmentation,
            NUM_CLASSES_SEGMENTATION,
            dataset_bands=mixed_dataset_bands,
            output_bands=mixed_output_bands,
            image_grep="input_data/*_img.tif",
            label_grep="label_data/*_label.tif",
            split=split_file_path,
        ), expected_mixed_filter_indices

    @pytest.mark.parametrize(
        "dataset",
        [
            "segmentation_dataset_with_HLS_bands",
            "segmentation_dataset_with_str_bands",
            "segmentation_dataset_with_interval_bands",
            "segmentation_dataset_with_str_bands",
            "segmentation_dataset_with_mixed_bands",
        ],
    )
    def test_correct_filter(self, dataset, request):
        fixture, expected = request.getfixturevalue(dataset)
        assert fixture.filter_indices == expected
