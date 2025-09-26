# Copyright contributors to the Terratorch project

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union
from typing_extensions import Self

from pydantic import BaseModel, model_validator

class PluginConfig(BaseModel):
    output_path: str = None
    """
    Default output folder path to be used when the out_data_format is set to path. 
    If omitted, the plugin will default to the current user home directory.
    """

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.output_path:
            self.output_path = str(Path.home())
        elif os.path.exists(self.output_path):
            if not os.access(self.output_path, os.W_OK):
                raise ValueError(f"The path: {self.output_path} is not writable")
        else:
            raise ValueError(f"The path: {self.output_path} does not exist")
                
        return self

class RequestData(BaseModel):

    data_format: Literal["b64_json", "path", "url"]
    """
    Data type for the input image.
    Allowed values are: [`b64_json`, `path`, `url`]
    """

    out_data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `url`]
    """

    data: Any
    """
    Input image data
    """

    indices: Optional[list[int]] = None


MultiModalPromptType = Union[RequestData]


class RequestOutput(BaseModel):
    
    data_format: Literal["b64_json", "path"]
    """
    Data type for the output image.
    Allowed values are: [`b64_json`, `path`]
    """

    data: Any
    """
    Output image data
    """

    request_id: Optional[str] = None
    """
    The vlLM request ID if applicable
    """
