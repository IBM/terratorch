from pathlib import Path
import logging
import warnings

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from torchgeo.trainers import BaseTask
from terratorch.models.utils import TemporalWrapper
from terratorch.registry import BACKBONE_REGISTRY

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
logging.basicConfig(level=logging.INFO)

class EmbeddingGenerationTask(BaseTask):
    """
    Task that runs inference over model backbone to generate and save embeddings.
    """

    def __init__(
            self,
            model: str,
            model_args: dict = None,
            output_dir: str = "embeddings",
            image_key: str = "image",
            filename_key: str = "filename",
            layers: list[int] = [-1],
            temporal_cfg: dict | None = None,
            output_format: str = "tiff",
    ) -> None:
        """Constructor for EmbeddingGenerationTask

        Args:
            model (str): Model name from backbone registry.
            model_args (dict, optional): Arguments passed to the model factory. Defaults to None.
            output_dir (str, optional): Directory to save embeddings. Defaults to "embeddings".
            image_key (str, optional): Key for image data in input data dictionary. Defaults to "image".
            filename_key (str, optional): Key for filename in input data dictionary. Defaults to "filename".
            layers (list[int], optional): List of layers to extract embeddings from. Defaults to [-1].
            temporal_cfg (dict, optional): Configuration for temporal processing. Defaults to None.
            output_format (str, optional): Format for saving embeddings ('tiff' for GeoTIFF, 'parquet' for GeoParquet). Defaults to "tiff".
        """

        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.temporal_cfg = temporal_cfg or {}

        super().__init__()
        self.save_hyperparameters()

    def configure_callbacks(self):
        return []
    
    def configure_models(self) -> None:
        """Instantiate backbone and optional temporal wrapper."""
        self.model = BACKBONE_REGISTRY.build(
            self.hparams.model,
            **(self.hparams.model_args or {}),
        )

        if self.temporal_cfg.get("temporal_wrapper", False):
            self.model = TemporalWrapper(
                self.model,
                pooling=self.temporal_cfg.get("pooling", "keep"),
            )

        self.model.eval()

    @torch.no_grad()
    def get_embeddings(
        self, 
        input: torch.Tensor | dict[str, torch.Tensor],
        layers: list[int] = [-1]
    ) -> tuple[list[torch.Tensor], list[int]]:
        """Run inference on the model and get selected layer outputs.

        Args:
            input (torch.Tensor | dict[str, torch.Tensor]): Input data to encode, format as expected by model backbone (Tensor or dict for multi-modal inputs).
            layers (list[int], optional): List of layers to extract embeddings from. Defaults to [-1]. Negative indices wrap around.

        Returns:
            tuple[list[torch.Tensor], list[int]]: Tuple containing list of embeddings and corresponding layer indices.
        """
        try:
            outputs = self.model(input)
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
        
        if not isinstance(outputs, list):
            logging.warning("Model output is not a list; assuming single output layer.")
            outputs = [outputs]

        n = len(outputs)
        layers = [l if l >= 0 else n + l for l in layers]
        valid_layers = [l for l in layers if 0 <= l < n]
        embeddings = [outputs[l] for l in valid_layers]

        return embeddings, valid_layers
    
    def pull_metadata(
        self, 
        data: dict
    ) -> dict:
        """Extract known metadata fields from `batch`, removing them from data and returning a metadata dict.
        Args:
            data (dict): Input data dictionary containing metadata.
        Returns:
            dict: Metadata dictionary.
        """
        def pop_first(d: dict, keys):
            for k in keys:
                if k in d:
                    return d.pop(k)
            return None
        
        # Aliases in priority order
        metadata_map = {
            "file_id":       ("file_id",),
            "product_id":    ("product_id",),
            "time":          ("time", "time_", "timestamp"),
            "grid_cell":     ("grid_cell",),
            "grid_row_u":    ("grid_row_u",),
            "grid_col_r":    ("grid_col_r",),
            "geometry":      ("geometry",),
            "utm_footprint": ("utm_footprint",),
            "crs":           ("crs", "utm_crs"),
            "pixel_bbox":    ("pixel_bbox",),
            "bounds":        ("bounds",),
            "center_lat":    ("center_lat", "centre_lat"),
            "center_lon":    ("center_lon", "centre_lon"),
        } 

        metadata = {}

        for key, aliases in metadata_map.items():
            value = pop_first(data, aliases)
            if value is not None:
                metadata[key] = value
        
        return metadata

    
    @torch.no_grad()
    def predict_step(self, batch: dict) -> None:
        image_key = self.hparams.image_key
        filename_key = self.hparams.filename_key

        if image_key not in batch:
            raise KeyError(f"Key '{image_key}' not found in input dictionary.")
        x = batch[image_key]

        if isinstance(x, dict) and filename_key in x:
            filenames = x.pop(filename_key)
            metadata = self.pull_metadata(x)
        else:
            filenames = batch.get(filename_key)
            if filenames is None:
                raise KeyError(f"Key '{filename_key}' not found in input dictionary.")
            if 'metadata' in batch:
                metadata = self.pull_metadata(batch['metadata'])
            else:   
                metadata = self.pull_metadata(batch)

        embeddings, layers = self.get_embeddings(x, self.hparams.layers)

        fmt = self.hparams.output_format.lower()
        if fmt not in ["tiff", "parquet"]:
            raise ValueError(f"Unsupported output format: {fmt}. Supported formats are 'tiff' and 'parquet'.")
        
        for embedding, layer in zip(embeddings, layers):
            self.save_embeddings(embedding, filenames, metadata, layer, fmt)

    def save_embeddings(
        self,
        embedding: torch.Tensor | dict[str, torch.Tensor],
        filenames: list[str] | None,
        metadata: dict,
        layer: int,
        fmt: str
    ) -> None:
        """Save embeddings for a given layer (per sample, optional per timestep and per modality)."""
        if isinstance(embedding, dict):
            for modality, t in embedding.items():
                path = self.output_path / f"layer_{layer}" / modality
                self.write_batch(t, filenames, metadata, path, fmt)
        elif isinstance(embedding, torch.Tensor):
            path = self.output_path / f"layer_{layer}"
            self.write_batch(embedding, filenames, metadata, path, fmt)
        else:
            raise TypeError(f"Unsupported embedding type: {type(embedding)}. Expected Tensor or dict of Tensors.")
        
    def write_batch(
            self,
            embedding: torch.Tensor,
            filenames: list[str],
            metadata: dict,
            dir_path: Path,
            fmt: str
    ) -> None:  
        """Write a batch (and optional timesteps) to GeoTIFF/GeoParquet."""
        dir_path.mkdir(parents=True, exist_ok=True)

        B = len(filenames)
        T = len(filenames[0]) if isinstance(filenames[0], (list, tuple, np.ndarray)) else None
    
        for b in range(B):
            if T is not None:
                for t in range(T):
                    filename = filenames[b][t]
                    metadata_sample = {k: v[b][t] for k, v in metadata.items()}
                    if fmt == "tiff":
                        self.write_tiff(embedding[b, t, ...], filename, metadata_sample, dir_path)
                    elif fmt == "parquet":
                        self.write_parquet(embedding[b, t, ...], filename, metadata_sample, dir_path)
            else:
                filename = filenames[b]
                metadata_sample = {k: v[b] for k, v in metadata.items()}
                if fmt == "tiff":
                    self.write_tiff(embedding[b, ...], filename, metadata_sample, dir_path)
                elif fmt == "parquet":
                    self.write_parquet(embedding[b, ...], filename, metadata_sample, dir_path)
    
    def write_tiff(
        self,
        embedding: torch.Tensor,
        filename: str,
        metadata: dict,
        dir_path: Path
        ) -> None:
        """Write a single sample to GeoTIFF."""
        filename = Path(filename).stem  
        out_path = dir_path / f"{Path(filename)}_embedding.tif"
        arr = embedding.detach().cpu().numpy()

        if arr.ndim == 2: # Add third dim for ViT outputs              
            arr = arr[None, ...]

        with rasterio.open( 
            out_path,
            "w",
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            count=arr.shape[0],
            dtype=arr.dtype
        ) as dst:
            dst.write(arr)

    def write_parquet(
        self,
        embedding: torch.Tensor,
        filename: list[str] | None,
        metadata: dict,
        dir_path: Path
    ) -> None:
        """Write a single sample to GeoParquet."""
        filename = Path(filename).stem  
        out_path = dir_path / f"{Path(filename)}_embedding.parquet"
        arr = embedding.detach().cpu().numpy()

        df = gpd.GeoDataFrame({
            'embedding': arr.tolist(),}
        )
        df.to_parquet(out_path)
