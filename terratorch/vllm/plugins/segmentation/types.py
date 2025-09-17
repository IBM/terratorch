# Copyright contributors to the Terratorch project

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union
from typing_extensions import Self

from pydantic import BaseModel, model_validator

class PluginConfig(BaseModel):
    output_path: str = None

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

class ImagePrompt(BaseModel):

    data_format: Literal["b64_json", "path", "url"]
    """
    This is the data type for the input image
    """

    image_format: str
    """
    This is the image format (e.g., jpeg, png, etc.)
    """

    out_data_format: Literal["b64_json", "path"]

    data: Any
    """
    Input image data
    """

    indices: Optional[list[int]] = None


MultiModalPromptType = Union[ImagePrompt]


class ImageRequestOutput(BaseModel):
    """
    The output data of an image request to vLLM. 

    Args:
        type (str): The data content type [path, object]
        format (str): The image format (e.g., jpeg, png, etc.)
        data (Any): The resulting data.
    """

    type: Literal["path", "b64_json"]
    format: str
    data: str
    request_id: Optional[str] = None
