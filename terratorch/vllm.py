
import torch
import torch.nn as nn
from terratorch.cli_tools import SemanticSegmentationTask
from typing import Optional
from collections.abc import Iterable,Iterator
import terratorch


class MultiModalDataGenerator():
    def __init__(self,config: dict):
        #print("DummyDataGenerator called")
        #self.task_type= config["task_args"]["task"] 
        self.task_type= config['model']["class_path"]
    
    def _get_mm_fields_config(self) -> dict[str:str]:
        if "SemanticSegmentationTasks"in self.task_type: 
            return {
                "pixel_values": "image",
                "location_coords": "image"
            }
        elif "WxCDownscalingTask" in self.task_type:
            return {
            "y": "image",
            "static_y": "image",
            "climate_y": "image",
            "x": "image",
            "static_x": "image",
            "climate_x": "image"
            }

class DummyDataGenerator():
    def __init__(self,config: dict):
        #print("DummyDataGenerator called")
        #self.task_type= config["task_args"]["task"] 
        self.task_type= config['model']["class_path"]
    
    def get_dummy_mm_data(self) -> dict[str,torch.Tensor]:

        if "SemanticSegmentationTask" in self.task_type:
            return {
            "pixel_values": torch.full((6, 512, 512), 1.0,
                                       dtype=torch.float16),
            "location_coords": torch.full((1, 2), 1.0, dtype=torch.float16),
        }
        elif "WxCDownscalingTask" in self.task_type:
            return {
            "y": torch.full((1, 360, 576), 1.0,dtype=torch.float16),
            "static_y": torch.full((11, 360, 576), 1.0,dtype=torch.float16),
            "climate_y": torch.full((1, 360, 576), 1.0,dtype=torch.float16),
            "x": torch.full((280, 60, 96), 1.0,dtype=torch.float16),
            "static_x": torch.full((11, 60, 96), 1.0,dtype=torch.float16),
            "climate_x": torch.full((140, 60, 96), 1.0,dtype=torch.float16),
            }
                
        else:
            raise Exception(f"task type {self.task_type} is not supported")
            

def lookup_task_name(class_path):
    if "SemanticSegmentationTask" in class_path:
        return 'segmentation'
    if "WxCDownscalingTask" in class_path:
        return 'WxCModelFactory'
    else:
        raise Exception("Factory not supported")
    
        
class InferenceRunner():
    
    def __init__(self, config: dict):
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_conf=  config["model"]
        self.task_type= config['model']["class_path"]
        task_name = lookup_task_name(model_conf["class_path"])
        model_factory = terratorch.registry.MODEL_FACTORY_REGISTRY.build(model_conf["init_args"]["model_factory"])
        self.model = model_factory.build_model(task=task_name,**model_conf['init_args']['model_args'])

#        self.task_type = config["task_args"]["task"]
#
#        if self.task_type == "SemanticSegmentationTask":
#
#            self.task = SemanticSegmentationTask(
#                config["model_args"],
#                config["task_args"]["model_factory"],
#                loss=config["task_args"]["loss"],
#                lr=config["task_args"]["lr"],
#                ignore_index=config["task_args"]["ignore_index"],
#                optimizer=config["task_args"]["optimizer"],
#                optimizer_hparams=config["optimizer_params"],
#                scheduler=config["task_args"]["scheduler"],
#                scheduler_hparams=config["scheduler_params"],
#                plot_on_val=config["task_args"]["plot_on_val"],
#                freeze_decoder=config["task_args"]["freeze_decoder"],
#                freeze_backbone=config["task_args"]["freeze_backbone"],
#            )
#
#        else:
#            raise ValueError(
#                "Unsupported task. "
#                "Only SemanticSegmentationTask is supported for now "
#                "by PrithviGeospatialMAE.")
    
    def _parse_and_validate_multimodal_data(
        self,**kwargs:object) -> tuple[torch.Tensor, Optional[torch.Tensor]] :


        if "SemanticSegmentationTask" in self.task_type:

            pixel_values = kwargs.pop("pixel_values", None)
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError(f"Incorrect type of pixel_values. "
                                f"Got type: {type(pixel_values)}")

            location_coords = kwargs.pop("location_coords", None)
            if not isinstance(location_coords, torch.Tensor):
                raise ValueError(f"Incorrect type of location_coords. "
                                f"Got type: {type(location_coords)}")
            location_coords = torch.unbind(location_coords, dim=0)[0]
            if location_coords.shape == torch.Size([0]):
                location_coords = None

            return pixel_values, location_coords
        elif "WxCDownscalingTask" in self.task_type:
            y= kwargs.pop("y", None)
            if not isinstance(y, torch.Tensor):
                raise ValueError(f"Incorrect type of y. "
                                f"Got type: {type(y)}")
            static_y= kwargs.pop("static_y", None)
            climate_y= kwargs.pop("climate_y", None)
            x= kwargs.pop("x", None)
            static_x= kwargs.pop("static_x", None)
            climate_x= kwargs.pop("climate_x", None)
            return { 
                    "y": y,
                    "static_y": static_y,
                    "climate_y": climate_y,
                    "x": x,
                    "static_x": static_x,
                    "climate_x": climate_x,
            }


        else:
            raise Exception(f"task type {self.task_type} is not supported")
    
    def forward(self,**kwargs: object):
        
        if "SemanticSegmentationTask" in self.task_type:
            pixel_values, location_coords = (
                self._parse_and_validate_multimodal_data(**kwargs))
            model_output = self.model(pixel_values,
                                           location_coords=location_coords)

        elif "WxCDownscalingTask" in self.task_type:
            input = self._parse_and_validate_multimodal_data(**kwargs)
            model_output = self.model(input)
        else:
            raise Exception(f"task type {self.task_type} is not supported")

        return model_output
    