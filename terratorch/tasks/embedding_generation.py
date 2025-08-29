from pathlib import Path
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
warnings.simplefilter("once", UserWarning)

class EmbeddingGenerationTask(BaseTask):
    """
    Task that runs inference over model backbone to generate and save embeddings.
    """

    def __init__(
            self,
            model: str,
            model_args: dict = None,
            output_dir: str = "embeddings",
            embed_file_key: str = "filename",
            layers: list[int] | None = None,
            temporal_cfg: dict | None = None,
            output_format: str = "tiff",
            has_cls: bool | None = None,
            embedding_pooling: str | None = None,
    ) -> None:
        """Constructor for EmbeddingGenerationTask

        Args:
            model (str): Model name from backbone registry.
            model_args (dict, optional): Arguments passed to the model factory. Defaults to None.
            output_dir (str, optional): Directory to save embeddings. Defaults to "embeddings".
            embed_file_key (str, optional): Identifier key for single file ids in input data, will be used as embedding identifiers. Defaults to "filename".
            layers (list[int], optional): List of layers to extract embeddings from. Defaults to [-1].
            temporal_cfg (dict, optional): Configuration for temporal processing. Defaults to None.
            output_format (str, optional): Format for saving embeddings ('tiff' for GeoTIFF, 'parquet' for GeoParquet). Defaults to "tiff".
            has_cls (bool | None, optional): Whether the model has a CLS token. Defaults to None.
            embedding_pooling (str | None, optional): Pooling method for embeddings. Defaults to None.
        """
        
        self.model = model
        self.model_args = model_args or {}
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.embed_file_key = embed_file_key
        self.layers = list(layers) if layers is not None else [-1]
        self.temporal_cfg = temporal_cfg or {}
        self.output_format = output_format.lower()
        self.has_cls = has_cls
        self.embedding_pooling = embedding_pooling

        if self.output_format not in ["tiff", "parquet"]:
            raise ValueError(f"Unsupported output format: {self.output_format}. Supported formats are 'tiff' and 'parquet'.")
        
        if self.embedding_pooling is not None:
            if self.has_cls is None and self.embedding_pooling.startswith("vit_"):
                warnings.warn("No 'has_cls' provided; assuming CLS if token count is odd.")
            if self.output_format == "tiff":
                warnings.warn("GeoTIFF output not recommended with embedding pooling, saves 1D vectors as (C,1,1).")
        else:
            warnings.warn(
                "GeoTIFF selected; 2D token embeddings (ViT) will be reshaped to "
                "[C, sqrt(num_tokens), sqrt(num_tokens)] after dropping CLS if present."
            )

        super().__init__()

    def infer_BT(self, x: torch.Tensor | dict[str, torch.Tensor]) -> tuple[int, int]:
        """Infer (B, T). For 5D assume [B, C, T, H, W] as standardized by TemporalWrapper."""
        if isinstance(x, dict):
            v = next(iter(x.values()))
        else:   
            v = x
        B = v.shape[0]
        T = v.shape[2] if v.ndim == 5 else 1 
        return B, T  

    def check_file_ids(
        self,
        file_ids: torch.Tensor | np.ndarray | list | tuple,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> None:
        """Validate `file_ids` matches (B,) or (B, T) inferred from `x`."""
        B, T = self.infer_BT(x)

        if isinstance(file_ids, (torch.Tensor, np.ndarray)):
            expected = (B,) if T == 1 else (B, T)
            if tuple(file_ids.shape) != expected:
                raise ValueError(f"`file_ids` shape mismatch: expected {expected}, got {tuple(file_ids.shape)}")
            return

        if isinstance(file_ids, (list, tuple)):
            if len(file_ids) != B:
                raise ValueError(f"`file_ids` length mismatch: expected {B}, got {len(file_ids)}")
            if T > 1 and isinstance(file_ids[0], (list, tuple, np.ndarray)) and len(file_ids[0]) != T:
                raise ValueError(f"`file_ids` must have inner length {T}, got {len(file_ids[0])}")
            return

        raise TypeError("`file_ids` must be a tensor/ndarray or a (nested) list/tuple")

    def configure_callbacks(self):
        return []
    
    def configure_models(self) -> None:
        """Instantiate backbone and optional temporal wrapper."""
        self.model = BACKBONE_REGISTRY.build(
            self.model,
            **(self.model_args or {}),
        )
        if self.temporal_cfg.get("temporal_wrapper", False):
            self.model = TemporalWrapper(
                self.model,
                pooling=self.temporal_cfg.get("temporal_pooling", "keep"),
            )

        self.model.eval()

    @torch.no_grad()
    def get_embeddings(
        self, 
        input: torch.Tensor | dict[str, torch.Tensor],
        layers: list[int]
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
            outputs = [outputs]

        n = len(outputs)
        layers = [l if l >= 0 else n + l for l in layers]
        for l in layers:
            if 0 > l or l >= n:
                raise IndexError(f"Layer index {l} out of bounds for model with {n} layer outputs.")
        
        embeddings = [outputs[l] for l in layers]
        return embeddings, layers
    
    @torch.no_grad()
    def predict_step(self, batch: dict) -> None:
        embed_file_key = self.embed_file_key
        x = batch['image']

        if isinstance(x, dict) and embed_file_key in x:
            file_ids = x.pop(embed_file_key)
            metadata = self.pull_metadata(x)
        else:
            file_ids = batch.get(embed_file_key)
            if file_ids is None:
                raise KeyError(f"Key '{embed_file_key}' not found in input dictionary.")
            if 'metadata' in batch:
                metadata = self.pull_metadata(batch['metadata'])
            else:   
                metadata = self.pull_metadata(batch)

        self.check_file_ids(file_ids, x)
        embeddings, layers = self.get_embeddings(x, self.layers)

        for embedding, layer in zip(embeddings, layers):
            self.save_embeddings(embedding, file_ids, metadata, layer)

    def save_embeddings(
        self,
        embedding: torch.Tensor | dict[str, torch.Tensor],
        file_ids: list[str] | None,
        metadata: dict,
        layer: int,
    ) -> None:
        """Save embeddings for a given layer (per sample, optional per timestep and per modality)."""
        if isinstance(embedding, dict):
            for modality, t in embedding.items():
                path = self.output_path / f"layer_{layer}" / modality
                self.write_batch(t, file_ids, metadata, path)
        elif isinstance(embedding, torch.Tensor):
            path = self.output_path / f"layer_{layer}"
            self.write_batch(embedding, file_ids, metadata, path)
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
            file_ids: list[str],
            metadata: dict,
            dir_path: Path,
    ) -> None:  
        """Write a batch (and optional timesteps) to GeoTIFF/GeoParquet."""
        dir_path.mkdir(parents=True, exist_ok=True)

        B = len(file_ids)
        T = len(file_ids[0]) if isinstance(file_ids[0], (list, tuple, np.ndarray)) else None
    
        for b in range(B):
            if T is not None:
                for t in range(T):
                    filename = file_ids[b][t]
                    metadata_sample = {k: v[b][t] for k, v in metadata.items()}
                    embedding_sample = self.pool_embedding(embedding[b, t, ...], self.embedding_pooling, self.has_cls)
                    if self.output_format == "tiff":
                        self.write_tiff(embedding_sample, filename, metadata_sample, dir_path)
                    elif self.output_format == "parquet":
                        self.write_parquet(embedding_sample, filename, metadata_sample, dir_path)
            else:
                filename = file_ids[b]
                metadata_sample = {k: v[b] for k, v in metadata.items()}
                embedding_sample = self.pool_embedding(embedding[b, ...], self.embedding_pooling, self.has_cls)
                if self.output_format == "tiff":
                    self.write_tiff(embedding_sample, filename, metadata_sample, dir_path)
                elif self.output_format == "parquet":
                    self.write_parquet(embedding_sample, filename, metadata_sample, dir_path)
    
    def pool_embedding(
        self,
        embedding: torch.Tensor,
        pooling: str | None,
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
            arr = arr.reshape(-1, 1, 1)
        elif arr.ndim == 2: 
            n_tokens, dim = arr.shape
            if self.has_cls is True or (self.has_cls is None and n_tokens % 2 == 1):
                arr = arr[1:, :]
                n_tokens -= 1
            s = int(np.sqrt(n_tokens))
            if s * s != n_tokens:
                raise ValueError(f"Cannot reshape {n_tokens} tokens into {s}x{s} grid.")
            arr = arr.reshape(s, s, dim).transpose(2, 0, 1)
   
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
            dst.update_tags(**{k: str(v) for k, v in metadata.items()})

    def write_parquet(
        self,
        embedding: torch.Tensor,
        filename: str,
        metadata: dict,
        dir_path: Path
    ) -> None:
        """Write a single sample to GeoParquet."""
        filename = Path(filename).stem  
        out_path = dir_path / f"{Path(filename)}_embedding.parquet"
        arr = embedding.detach().cpu().numpy()

        row = {"embedding": arr.tolist()}
        row.update({k: (v.tolist() if v.ndim else v.item()) for k, v in metadata.items()})

        df = gpd.GeoDataFrame([row])  
        df.to_parquet(out_path, index=False)