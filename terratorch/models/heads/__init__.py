# Copyright contributors to the Terratorch project

from terratorch.models.heads.scalar_head import ScalarHead
from terratorch.models.heads.regression_head import RegressionHead
from terratorch.models.heads.segmentation_head import SegmentationHead

# TODO: Remove in a version v1.3 or later
# Deprecated alias for backward compatibility
ClassificationHead = ScalarHead

__all__ = ["ScalarHead", "RegressionHead", "SegmentationHead"]
