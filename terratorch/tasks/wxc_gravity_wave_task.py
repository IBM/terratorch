

from torchgeo.trainers import BaseTask
import torch.nn as nn

class WxCGravityWaveTask(BaseTask):
    def __init__(self, model_factory):
        self.model_factory = model_factory
        super().__init__()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def configure_models(self):
        self.model = self.model_factory.build_model(backbone='gravitywave', aux_decoders=None)