import torch
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY 

@TERRATORCH_BACKBONE_REGISTRY.register
def alexnet_encoder(num_channels:int=2, pretrained:bool=False):

    backbone = AlexNetEncoder(num_channels=num_channels)
    if pretrained:
        print("This is my life")

    return backbone

@TERRATORCH_BACKBONE_REGISTRY.register
def alexnet(num_channels:int=2, pretrained:bool=False):

    backbone = AlexNet(num_channels=num_channels)
    if pretrained:
        print("This is my life")

    return backbone

class AlexNetEncoder(torch.nn.Module):

    def __init__(self, num_channels:int=3):

        super(AlexNetEncoder, self).__init__()

        self.num_channels = num_channels
        self.first_level_num_channels = 96
        self.second_level_num_channels = 256
        self.third_level_num_channels = 384
        self.fourth_level_num_channels = 4096

        self.first_layer_kernel_size = 11
        self.third_layer_kernel_size = 5
        self.default_kernel_size = 3

        self.first_layer_stride = 4
        self.default_stride = 2

        self.third_layer_pad = 2
        self.default_pad = 1

        self.pipeline = torch.nn.Sequential(
                                     torch.nn.Conv2d(self.num_channels,
                                                     self.first_level_num_channels,
                                                     kernel_size=self.first_layer_kernel_size,
                                                     stride=self.first_layer_stride),
                                     torch.nn.MaxPool2d(kernel_size=self.default_kernel_size,
                                                        stride=self.default_stride),
                                     torch.nn.Conv2d(self.first_level_num_channels,
                                                     self.second_level_num_channels,
                                                     kernel_size=self.third_layer_kernel_size,
                                                     padding=self.third_layer_pad),
                                     torch.nn.MaxPool2d(kernel_size=self.default_kernel_size,
                                                        stride=self.default_stride),
                                     torch.nn.Conv2d(self.second_level_num_channels,
                                                        self.second_level_num_channels,
                                                        kernel_size=self.default_kernel_size,
                                                        padding=self.default_pad),
                                     torch.nn.Conv2d(self.second_level_num_channels,
                                                        self.second_level_num_channels,
                                                        kernel_size=self.default_kernel_size,
                                                        padding=self.default_pad),
                                     torch.nn.Conv2d(self.second_level_num_channels,
                                                        self.third_level_num_channels,
                                                        kernel_size=self.default_kernel_size,
                                                        padding=self.default_pad),
                                     torch.nn.MaxPool2d(kernel_size=self.default_kernel_size,
                                                        stride=self.default_stride),
                                    )
        self.out_channels = 8*[self.third_level_num_channels]
    def forward(self, x):

        return [self.pipeline(x)[:, ...]]

@TERRATORCH_BACKBONE_REGISTRY.register
class AlexNet(torch.nn.Module):

    def __init__(self, num_classes:int=10, num_channels:int=3):

        super(AlexNet, self).__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.first_level_num_channels = 96
        self.second_level_num_channels = 256
        self.third_level_num_channels = 384
        self.fourth_level_num_channels = 4096

        self.first_layer_kernel_size = 11
        self.third_layer_kernel_size = 5
        self.default_kernel_size = 3

        self.first_layer_stride = 4
        self.default_stride = 2

        self.third_layer_pad = 2
        self.default_pad = 1

        self.pipeline = torch.nn.Sequential(
                                     torch.nn.Conv2d(self.num_channels,
                                                     self.first_level_num_channels,
                                                     kernel_size=self.first_layer_kernel_size,
                                                     stride=self.first_layer_stride),
                                     torch.nn.MaxPool2d(kernel_size=self.default_kernel_size,
                                                        stride=self.default_stride),
                                     torch.nn.Conv2d(self.first_level_num_channels,
                                                     self.second_level_num_channels,
                                                     kernel_size=self.third_layer_kernel_size,
                                                     padding=self.third_layer_pad),
                                     torch.nn.MaxPool2d(kernel_size=self.default_kernel_size,
                                                        stride=self.default_stride),
                                     torch.nn.Conv2d(self.second_level_num_channels,
                                                        self.second_level_num_channels,
                                                        kernel_size=self.default_kernel_size,
                                                        padding=self.default_pad),
                                     torch.nn.Conv2d(self.second_level_num_channels,
                                                        self.second_level_num_channels,
                                                        kernel_size=self.default_kernel_size,
                                                        padding=self.default_pad),
                                     torch.nn.Conv2d(self.second_level_num_channels,
                                                        self.third_level_num_channels,
                                                        kernel_size=self.default_kernel_size,
                                                        padding=self.default_pad),
                                     torch.nn.MaxPool2d(kernel_size=self.default_kernel_size,
                                                        stride=self.default_stride),
                                     torch.nn.Flatten(),
                                     torch.nn.Linear(25*self.third_level_num_channels,
                                                     self.fourth_level_num_channels),
                                     torch.nn.Linear(self.fourth_level_num_channels,
                                                     self.num_classes)
                                    )

        self.out_channels = 8*[self.third_level_num_channels]

    def forward(self, x):

        return self.pipeline(x)


