import pytest
import torch

from terratorch.models.backbones.terramind.model.encoder_embeddings import (
    SequenceEmbEncoderEmbedding,  # Replace 'your_module' with the actual module path
)


@pytest.fixture
def sequence_emb_encoder():
    return SequenceEmbEncoderEmbedding(max_length=10, dim_tokens=768)


def test_sequence_emb_encoder_embedding_forward(sequence_emb_encoder):
    # Prepare input data
    orig_emb = torch.rand((2, 10, 4096))  # (B, L, E)
    input_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # (B, L)

    # Call the forward method
    d = {"tensor": orig_emb, "input_mask": input_mask}
    output = sequence_emb_encoder(d)

    # Check the output shape and values
    assert output["x"].shape == (2, 10, 768)
    assert output["emb"].shape == (2, 10, 768)


def test_sequence_emb_encoder_embedding_no_weight_decay():
    sequence_emb_encoder = SequenceEmbEncoderEmbedding(max_length=10, dim_tokens=768)
    assert sequence_emb_encoder.no_weight_decay() == set()
