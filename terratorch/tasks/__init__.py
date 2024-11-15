from terratorch.tasks.classification_tasks import ClassificationTask
from terratorch.tasks.regression_tasks import PixelwiseRegressionTask
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask
from terratorch.tasks.multilabel_classification_tasks import MultiLabelClassificationTask
try:
    wxc_present = True
    from terratorch.tasks.wxc_downscaling_task import WxCDownscalingTask 
except ImportError as e:
    print('wxc_downscaling not installed')
    wxc_present = False


__all__ = (
    "SemanticSegmentationTask",
    "PixelwiseRegressionTask",
    "ClassificationTask",
    "MultiLabelClassificationTask"
    "BATCH_IDX_FOR_VALIDATION_PLOTTING",
)

if wxc_present:
    __all__.__add__(("WxCDownscalingTask", ))