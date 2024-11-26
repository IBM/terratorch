

from torchgeo.trainers import BaseTask
import torch.nn as nn
import torch

class WxCGravityWaveTask(BaseTask):
    def __init__(self, model_factory, mode, learning_rate=0.1):
        if mode not in ['train', 'eval']:
            raise ValueError(f'mode {mode} is not supported. (train, eval)')
        self.model_factory = model_factory
        self.learning_rate = learning_rate
        super().__init__()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def configure_models(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_factory.build_model(backbone='prithviwxc', aux_decoders=None)
        self.model = self.model.to(device)

    def training_step(self, batch, batch_idx):
        output: torch.Tensor = self.model(batch)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        