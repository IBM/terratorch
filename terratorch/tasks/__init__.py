import logging
from terratorch.tasks.classification_tasks import ClassificationTask
from terratorch.tasks.regression_tasks import PixelwiseRegressionTask
from terratorch.tasks.segmentation_tasks import SemanticSegmentationTask
from terratorch.tasks.multilabel_classification_tasks import MultiLabelClassificationTask
from terratorch.tasks.base_task import TerraTorchTask
try:
    wxc_present = True
    from terratorch.tasks.wxc_downscaling_task import WxCDownscalingTask 
    from terratorch.tasks.wxc_task import WxCTask
    logging.getLogger('terratorch').debug('wxc_downscaling found.')
except ImportError as e:
    import logging
    logging.getLogger('terratorch').debug('wxc_downscaling not installed')
    wxc_present = False


__all__ = (
    "SemanticSegmentationTask",
    "PixelwiseRegressionTask",
    "ClassificationTask",
    "MultiLabelClassificationTask"
    "BATCH_IDX_FOR_VALIDATION_PLOTTING",
)

if wxc_present:
    __all__.__add__(("WxCDownscalingTask", "WxCTask",))
