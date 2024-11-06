

from torchgeo.trainers import BaseTask
import torch.nn as nn
import torch

class WxCGravityWaveTask(BaseTask):
    def __init__(self, model_factory, learning_rate=0.1):
        self.model_factory = model_factory
        self.learning_rate = learning_rate
        super().__init__()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def configure_models(self):
        self.model = self.model_factory.build_model(backbone='gravitywave', aux_decoders=None)

    def training_step(self, *args, **kwargs):
        None
        
    def train_dataloader(self):
        None
        