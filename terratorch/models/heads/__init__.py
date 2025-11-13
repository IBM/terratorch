# Copyright contributors to the Terratorch project
import warnings
from terratorch.models.heads.scalar_head import ScalarHead
from terratorch.models.heads.regression_head import RegressionHead
from terratorch.models.heads.segmentation_head import SegmentationHead

# TODO: Remove in a version v1.3 or later
warnings.warn("ClassificationHead is deprecated. Use ScalarHead instead", DeprecationWarning)
ClassificationHead = ScalarHead

__all__ = ["ScalarHead", "RegressionHead", "SegmentationHead"]
