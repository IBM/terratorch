import numpy as np
import torch
from terratorch.datasets.transforms import MinMaxNormalize

def test_minmax_normalize_numpy_channel_last():
    means = [10.0, 20.0]
    stds = [5.0, 5.0]
    transform = MinMaxNormalize(means=means, stds=stds, scale=2.0, channel_axis=-1)
    image = np.stack(
        (
            np.full((224, 224), means[0], dtype=np.float32),
            np.full((224, 224), means[1], dtype=np.float32),
        ),
        axis=-1,
    )
    normalized = transform.apply(image)
    assert np.allclose(normalized[..., 0], 0.5)
    assert np.allclose(normalized[..., 1], 0.5)


def test_minmax_normalize_torch_channel_first():
    means = [1.0, 3.0, 5.0]
    stds = [1.0, 1.0, 1.0]
    transform = MinMaxNormalize(means=means, stds=stds, scale=2.0, channel_axis=0)
    tensor = torch.tensor(means, dtype=torch.float32).view(3, 1, 1).expand(3, 224, 224)
    normalized = transform.apply(tensor)
    expected = torch.full_like(normalized[0], 0.5)
    assert torch.allclose(normalized[0], expected)
    assert torch.allclose(normalized[1], expected)
    assert torch.allclose(normalized[2], expected)

if __name__ == "__main__":
    test_minmax_normalize_numpy_channel_last()
    test_minmax_normalize_torch_channel_first()