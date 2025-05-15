
import torch
from torch import nn
from tokenizers import Tokenizer


class CaptionTokenizer(nn.Module):
    def __init__(self, tokenizer_file, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_tokenizer = Tokenizer.from_file(tokenizer_file)
        self.text_tokenizer.enable_padding()

    def encode(self, text: list[str], device: torch.device, *args, **kwargs) -> tuple[torch.Tensor]:
        """
        Encodes coords to token ids. Returns tuple to be compatible with image tokenizers.

        Args:
            coords (torch.Tensor): Center coordinates of image with shape [B, 2] with [lon, lat] values in second dim.

        Returns:
            tok_ids(tuple[torch.Tensor]): Token ids with shape [B, 2]
        """
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(text, add_special_tokens=True)]
        tok_ids = torch.tensor(tok_ids, device=device)
        # Returns tuple to be compatible with image tokenizers
        return (tok_ids,)


class CoordsTokenizer(nn.Module):
    def __init__(self, tokenizer_file, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_tokenizer = Tokenizer.from_file(tokenizer_file)

    def encode(self, coords: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor]:
        """
        Encodes coords to token ids. Returns tuple to be compatible with image tokenizers.

        Args:
            coords (torch.Tensor): Center coordinates of image with shape [B, 2] with [lon, lat] values in second dim.

        Returns:
            tok_ids(tuple[torch.Tensor]): Token ids with shape [B, 2]
        """
        if coords.shape[1] != 2:
            raise ValueError(f'Expect coords data in shape [batch, 2] with [lon, lat] values, '
                             f'got coords with shape {coords.shape}.')

        # Align coords with 0.25 degree grid
        coords = (coords * 4).round() / 4

        def coords_to_tok(lonlat: torch.Tensor) -> torch.Tensor:
            lon_tok = self.text_tokenizer.token_to_id(f"lon={lonlat[0].item():.2f}")
            lat_tok = self.text_tokenizer.token_to_id(f"lat={lonlat[1].item():.2f}")
            return torch.tensor([lat_tok, lon_tok], device=lonlat.device)

        tok_ids = torch.stack([coords_to_tok(sample) for sample in coords])

        # Returns tuple to be compatible with image tokenizers
        return (tok_ids,)
