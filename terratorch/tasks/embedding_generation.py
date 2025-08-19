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
            embedding_pooling: str | None = None,
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
            embedding_pooling (str | None, optional): Pooling method for embeddings. Defaults to None.
        """
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.temporal_cfg = temporal_cfg or {}

        super().__init__()
        self.save_hyperparameters()

        self._warned_keys: set[str] = set() 

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned_keys:
            logging.warning(msg)
            self._warned_keys.add(key)

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
                    embedding_sample = self.pool_embedding(embedding[b, t, ...], self.hparams.get("embedding_pooling"), self.hparams.get("has_cls", None))
                    if fmt == "tiff":
                        self.write_tiff(embedding_sample, filename, metadata_sample, dir_path)
                    elif fmt == "parquet":
                        self.write_parquet(embedding_sample, filename, metadata_sample, dir_path)
            else:
                filename = filenames[b]
                metadata_sample = {k: v[b] for k, v in metadata.items()}
                embedding_sample = self.pool_embedding(embedding[b, ...], self.hparams.get("embedding_pooling"), self.hparams.get("has_cls", None))
                if fmt == "tiff":
                    self.write_tiff(embedding_sample, filename, metadata_sample, dir_path)
                elif fmt == "parquet":
                    self.write_parquet(embedding_sample, filename, metadata_sample, dir_path)
    
    def pool_embedding(
        self,
        embedding: torch.Tensor,
        pooling: str,
        has_cls: bool | None,
    ) -> torch.Tensor:
        """Apply pooling to embeddings."""
        if pooling in (None, "None", "keep"):
            return embedding

        if pooling.startswith("vit_"):
            if embedding.dim() != 2:
                raise ValueError(f"Expected 2D embedding for ViT pooling, got {embedding.dim()}D.")
            if has_cls is None:
                has_cls = embedding.shape[0] % 2 == 1
            if has_cls is False and pooling == "vit_cls":
                raise ValueError("Cannot use 'vit_cls' pooling without a CLS token.")
            if has_cls is True and pooling != "vit_cls":
                embedding = embedding[1:, :]

        if pooling.startswith("cnn_") and embedding.dim() != 3:
            raise ValueError(f"Expected 3D embedding for CNN pooling, got {embedding.dim()}D.")

        if pooling == "vit_mean":
            return embedding.mean(dim=0)
        elif pooling == "vit_max":
            return embedding.max(dim=0).values
        elif pooling == "vit_min":
            return embedding.min(dim=0).values
        elif pooling == "vit_cls":
            return embedding[0, :]
        elif pooling == "cnn_mean":
            return embedding.mean(dim=(1, 2))
        elif pooling == "cnn_max":
            return embedding.max(dim=(1, 2)).values
        elif pooling == "cnn_min":
            return embedding.min(dim=(1, 2)).values
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}.")
        
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

        if arr.ndim == 1:
            self._warn_once(
                "tiff 1d",
                "1D embedding detected; GeoTIFF not recommended. Saving with height=1 and width=1."
            )
            arr = arr.reshape(1, 1, -1)
        elif arr.ndim == 2:    
            self._warn_once(
                "tiff 2d",
                "2D embedding detected with GeoTIFF output selected. "
                "Assuming token sequence and reshaping to "
                "[embedding_size, sqrt(num_tokens), sqrt(num_tokens)], "
                "ignoring the CLS token if present. "
                "Consider using 'parquet' for saving 2D embeddings."
            )
            if arr.shape[0] % 2 == 1:
                arr = arr[1:]
            sqrt_size = int(np.sqrt(arr.shape[0]))
            arr = arr.reshape(arr.shape[1], sqrt_size, sqrt_size)
                
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