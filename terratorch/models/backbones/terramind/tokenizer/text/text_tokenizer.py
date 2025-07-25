import warnings

import torch
from torch import nn
from tokenizers import Tokenizer


class CaptionTokenizer(nn.Module):
    def __init__(self, tokenizer_file, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_tokenizer = Tokenizer.from_file(tokenizer_file)
        self.text_tokenizer.enable_padding()

    def encode(self, text: list[str], device: torch.device, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Args:
            text list[str]: Text to be tokenized
            device: torch.device
        Returns:
            dict for generation sampler input
        """
        # Add start token
        text = [t + " [S_1]" for t in text]

        # Tokenize
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(text, add_special_tokens=True)]

        # Add end token
        eos_id = self.text_tokenizer.encode("[EOS]").ids
        tok_ids = [t + eos_id for t in tok_ids]

        tok_ids = torch.tensor(tok_ids, device=device)

        text_dict = {
            "tensor": tok_ids,
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }

        return text_dict

    def decode_text(self, mod_dict, key="caption"):
        """
        Decodes a text sequence from a model dictionary.

        Args:
            mod_dict (dict): Model output dictionary.
            key (str): Key of the text modality to decode.
        """
        decoded_texts = []

        for i in range(mod_dict[key]["tensor"].shape[0]):
            seq = mod_dict[key]["tensor"][i]
            seq = seq[mod_dict[key]["input_mask"][i] == 0]
            seq = seq.tolist()

            merged_text = self.text_tokenizer.decode(seq, skip_special_tokens=False)

            decoded_texts.append(merged_text.replace(" [EOS]", ""))

        return decoded_texts


class CoordsTokenizer(nn.Module):
    def __init__(self, tokenizer_file, pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_tokenizer = Tokenizer.from_file(tokenizer_file)

    def encode(self, coords: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Encodes coords to token ids. Returns tuple to be compatible with image tokenizers.

        Args:
            coords (torch.Tensor): Center coordinates of image with shape [B, 2] with [lon, lat] values in second dim.

        Returns:
            tok_ids(tuple[torch.Tensor]): Token ids with shape [B, 2]
        """
        if coords.shape[1] != 2:
            raise ValueError(f"Expect coords data in shape [batch, 2] with [lon, lat] values, "
                             f"got coords with shape {coords.shape}.")

        # Align coords with 0.25 degree grid
        coords = (coords * 4).round() / 4
        device = coords.device

        coords = [f"lat={c[1].item():.2f} lon={c[0].item():.2f} [EOS]" for c in coords]

        # Tokenize
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(coords, add_special_tokens=True)]

        tok_ids = torch.tensor(tok_ids, device=device)

        coords_dict = {
            "tensor": tok_ids,
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }

        return coords_dict

    def decode_text(self, mod_dict, key="coords"):
        """
        Decodes a coordinate sequence from a modality dictionary.

        Args:
            mod_dict (dict): Model output dictionary.
            key (str): Key of the coords modality to decode.
        """
        coords = []

        B = mod_dict[key]["tensor"].shape[0]

        for i in range(B):
            seq = mod_dict[key]["tensor"][i].tolist()[:2]

            text = self.text_tokenizer.decode(seq, skip_special_tokens=False)

            lat, lon = text.split(" ")
            if lat.startswith("lon") or lon.startswith("lat"):
                warnings.warn(f"Coordinate generation did not work correctly, generated text: {text}. Returning NaN.")
                coords.append([torch.nan, torch.nan])
                continue

            coords.append([float(lon.strip("lon=")), float(lat.strip("lat="))])

        return coords
