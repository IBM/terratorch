import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A  # noqa: N812
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import pad_numpy, to_tensor

CAT_TILES = ["31TBF", "31TCF", "31TCG", "31TDF", "31TDG"]
FR_TILES = ["31TCJ", "31TDK", "31TCL", "31TDM", "31UCP", "31UDR"]

MAX_TEMPORAL_IMAGE_SIZE = (366, 366)

SELECTED_CLASSES = [
    110,   # 'Wheat'
    120,   # 'Maize'
    140,   # 'Sorghum'
    150,   # 'Barley'
    160,   # 'Rye'
    170,   # 'Oats'
    330,   # 'Grapes'
    435,   # 'Rapeseed'
    438,   # 'Sunflower'
    510,   # 'Potatoes'
    770,   # 'Peas'
]


class Sen4AgriNet(NonGeoDataset):
    def __init__(
        self,
        data_root: str,
        bands: list[str] | None = None,
        scenario: str = "random",
        split: str = "train",
        transform: A.Compose = None,
        truncate_image: int | None = None,
        pad_image: int | None = None,
        spatial_interpolate_and_stack_temporally: bool = False,  # noqa: FBT001, FBT002
        seed: int = 42,
        fixed_time_window: bool = True,
        start_month: int = 4,
        end_month: int = 9,
        output_size: tuple | None = MAX_TEMPORAL_IMAGE_SIZE,
        requires_norm: bool = True,
        normalization_div: float = 10000.0,
        use_linear_encoder: bool = True,
    ):
        """
        Pytorch Dataset class to load samples from the Sen4AgriNet dataset, supporting
        multiple scenarios for splitting the data.

        Args:
            data_root (str): Root directory of the dataset.
            bands (list of str, optional): List of band names to load. Defaults to all available bands.
            scenario (str): Defines the splitting scenario to use. Options are:
                - 'random': Random split of the data.
                - 'spatial': Split by geographical regions (Catalonia and France).
                - 'spatio-temporal': Split by region and year (France 2019 and Catalonia 2020).
            split (str): Specifies the dataset split. Options are 'train', 'val', or 'test'.
            transform (albumentations.Compose, optional): Albumentations transformations to apply to the data.
            truncate_image (int, optional): Number of timesteps to truncate the time dimension of the image.
                If None, no truncation is applied. Default is None.
            pad_image (int, optional): Number of timesteps to pad the time dimension of the image.
                If None, no padding is applied. Default is None.
            spatial_interpolate_and_stack_temporally (bool): Whether to interpolate bands and concatenate them over time.
            seed (int): Random seed used for data splitting.
            fixed_time_window (bool): Whether to use a fixed time window from start_month to end_month.
            start_month (int): Starting month for the fixed time window (inclusive). Default is 4 (April).
            end_month (int): Ending month for the fixed time window (inclusive). Default is 9 (September).
            output_size (tuple of ints, optional): Size (height, width) of the subpatches.
            requires_norm (bool): Whether to normalize the data to [0, 1] range.
            normalization_div (float): Value to divide the data for normalization. Default is 10000.0.
        """
        self.data_root = Path(data_root) / "data"
        self.transform = transform if transform else lambda **batch: to_tensor(batch)
        self.scenario = scenario
        self.seed = seed
        self.truncate_image = truncate_image
        self.pad_image = pad_image
        self.spatial_interpolate_and_stack_temporally = spatial_interpolate_and_stack_temporally
        self.fixed_time_window = fixed_time_window
        self.start_month = start_month
        self.end_month = end_month
        self.output_size = output_size
        self.requires_norm = requires_norm
        self.normalization_div = normalization_div
        self.use_linear_encoder = use_linear_encoder

        if bands is None:
            bands = ["B01", "B02", "B03", "B04", "B05", "B06",
                     "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
        self.bands = bands

        self.image_files = list(self.data_root.glob("**/*.nc"))

        self.train_files, self.val_files, self.test_files = self.split_data()

        if split == "train":
            self.image_files = self.train_files
        elif split == "val":
            self.image_files = self.val_files
        elif split == "test":
            self.image_files = self.test_files

        if self.use_linear_encoder:
            self.linear_encoder = {val: i + 1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
            self.linear_encoder[0] = 0

    def __len__(self):
        return len(self.image_files)

    def split_data(self):
        random.seed(self.seed)

        if self.scenario == "random":
            random.shuffle(self.image_files)
            total_files = len(self.image_files)
            train_split = int(0.6 * total_files)
            val_split = int(0.8 * total_files)

            train_files = self.image_files[:train_split]
            val_files = self.image_files[train_split:val_split]
            test_files = self.image_files[val_split:]

        elif self.scenario == "spatial":
            catalonia_files = [f for f in self.image_files if any(tile in f.stem for tile in CAT_TILES)]
            france_files = [f for f in self.image_files if any(tile in f.stem for tile in FR_TILES)]

            val_split_cat = int(0.2 * len(catalonia_files))
            train_files = catalonia_files[val_split_cat:]
            val_files = catalonia_files[:val_split_cat]
            test_files = france_files

        elif self.scenario == "spatio-temporal":
            france_files = [f for f in self.image_files if any(tile in f.stem for tile in FR_TILES)]
            catalonia_files = [f for f in self.image_files if any(tile in f.stem for tile in CAT_TILES)]

            france_2019_files = [f for f in france_files if "2019" in f.stem]
            catalonia_2020_files = [f for f in catalonia_files if "2020" in f.stem]

            val_split_france_2019 = int(0.2 * len(france_2019_files))
            train_files = france_2019_files[val_split_france_2019:]
            val_files = france_2019_files[:val_split_france_2019]
            test_files = catalonia_2020_files

        return train_files, val_files, test_files

    def __getitem__(self, index: int):
        patch_file = self.image_files[index]

        with h5py.File(patch_file, "r") as patch_data:
            output = {}
            band_medians_per_band = []
            for band in self.bands:
                band_group = patch_data[band]
                band_data = band_group[f"{band}"][:]
                time_vector = band_group["time"][:]

                # Convert time_vector to datetime objects
                time_vector = [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_vector]

                # Filter data between start_month and end_month if fixed_time_window is True
                if self.fixed_time_window:
                    month_mask = [(self.start_month <= dt.month <= self.end_month) for dt in time_vector]
                    band_data = band_data[month_mask]
                    time_vector = [dt for dt, mask in zip(time_vector, month_mask, strict=False) if mask]

                # If no data left after filtering, continue to next band
                if len(band_data) == 0:
                    # Handle missing data by filling with zeros
                    num_months = self.end_month - self.start_month + 1
                    median_data = [np.zeros((band_data.shape[1], band_data.shape[2]))] * num_months
                else:
                    # Group data by month and compute medians
                    band_data_by_month = {}
                    for data, dt in zip(band_data, time_vector, strict=False):
                        month = dt.month
                        if month not in band_data_by_month:
                            band_data_by_month[month] = []
                        band_data_by_month[month].append(data)

                    # Compute median for each month
                    median_data_by_month = {}
                    for month in band_data_by_month:
                        median_data_by_month[month] = np.median(band_data_by_month[month], axis=0)

                    # Handle missing months
                    months = list(range(self.start_month, self.end_month + 1))
                    median_data = []
                    for month in months:
                        if month in median_data_by_month:
                            median_data.append(median_data_by_month[month])
                        else:
                            # Interpolate or replicate data
                            prev_month = month - 1
                            next_month = month + 1
                            prev_data = None
                            next_data = None
                            while prev_month >= self.start_month:
                                if prev_month in median_data_by_month:
                                    prev_data = median_data_by_month[prev_month]
                                    break
                                prev_month -= 1
                            while next_month <= self.end_month:
                                if next_month in median_data_by_month:
                                    next_data = median_data_by_month[next_month]
                                    break
                                next_month += 1
                            if prev_data is not None and next_data is not None:
                                interpolated_data = (prev_data + next_data) / 2
                                median_data.append(interpolated_data)
                            elif prev_data is not None:
                                median_data.append(prev_data)
                            elif next_data is not None:
                                median_data.append(next_data)
                            else:
                                # No data available, fill with zeros
                                median_data.append(np.zeros((band_data.shape[1], band_data.shape[2])))

                # Stack median_data into an array of shape (num_months, H, W)
                median_data = np.stack(median_data, axis=0)  # Shape: (num_months, H, W)

                # Interpolate spatially if required
                if self.spatial_interpolate_and_stack_temporally:
                    median_data = torch.from_numpy(median_data)
                    median_data = median_data.clone().detach()

                    interpolated = F.interpolate(
                        median_data.unsqueeze(0), size=MAX_TEMPORAL_IMAGE_SIZE, mode="bilinear", align_corners=False
                    ).squeeze(0)
                    band_medians_per_band.append(interpolated.numpy())
                else:
                    band_medians_per_band.append(median_data)

            # Stack over bands and transpose to shape (num_months, num_bands, H, W)
            images = np.stack(band_medians_per_band, axis=1)  # Shape: (num_months, num_bands, H, W)

            # Apply normalization if required
            if self.requires_norm:
                images = images / self.normalization_div

            # Convert to torch.Tensor
            images = torch.from_numpy(images).float()
            # Handle images of different sizes with padding and subpatching
            if self.output_size is not None:
                images, num_subpatches_h, num_subpatches_w = self.apply_padding_and_subpatching(images)
                # Select subpatch based on index
                subpatch_idx = index % (num_subpatches_h * num_subpatches_w)
                images = self.get_subpatch(images, subpatch_idx, num_subpatches_h, num_subpatches_w)

            # Prepare the output
            output["image"] = images  # Shape: (num_months, num_bands, H_sub, W_sub)

            # Load labels and parcels
            labels = patch_data["labels"]["labels"][:].astype(int)
            parcels = patch_data["parcels"]["parcels"][:].astype(int)

            # Apply same padding and subpatching to labels
            if self.output_size is not None:
                labels = self.apply_padding(labels)
                labels = self.get_subpatch(labels, subpatch_idx, num_subpatches_h, num_subpatches_w)
                labels = labels.long()

            output["mask"] = labels

            if self.use_linear_encoder:
                output["mask"] = self.map_mask_to_discrete_classes(output["mask"], self.linear_encoder)

            if self.transform:
                transformed_images = []
                for i in range(images.shape[0]):
                    img = images[i].permute(1, 2, 0).numpy()  # (num_bands, H_sub, W_sub) -> (H_sub, W_sub, num_bands)
                    transformed = self.transform(image=img)
                    img = transformed["image"]
                    transformed_images.append(img)
                images = torch.stack(transformed_images, dim=0)  # (num_months, num_bands, H_sub, W_sub)
                output["image"] = images

            output["parcels"] = torch.from_numpy(parcels).long()

        return output

    def apply_padding_and_subpatching(self, images):
        """
        Apply padding to images and compute number of subpatches.
        """
        _, _, height, width = images.shape  # images shape: (num_months, num_bands, H, W)
        pad_h = (self.output_size[0] - height % self.output_size[0]) % self.output_size[0]
        pad_w = (self.output_size[1] - width % self.output_size[1]) % self.output_size[1]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        images_padded = F.pad(
            images,
            pad=(pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0
        )

        padded_height = images_padded.shape[2]
        padded_width = images_padded.shape[3]
        num_subpatches_h = padded_height // self.output_size[0]
        num_subpatches_w = padded_width // self.output_size[1]

        return images_padded, num_subpatches_h, num_subpatches_w

    def get_subpatch(self, data, subpatch_idx, num_subpatches_h, num_subpatches_w):
        idx_h = subpatch_idx // num_subpatches_w
        idx_w = subpatch_idx % num_subpatches_w
        start_h = idx_h * self.output_size[0]
        start_w = idx_w * self.output_size[1]
        end_h = start_h + self.output_size[0]
        end_w = start_w + self.output_size[1]

        if data.ndim == 4:
            # Imagens: (num_months, num_bands, H, W)
            return data[:, :, start_h:end_h, start_w:end_w]
        elif data.ndim == 2:
            # Labels: (H, W)
            return data[start_h:end_h, start_w:end_w]
        else:
            msg = f"Unsupported data dimensions: {data.ndim}"
            raise ValueError(msg)

    def apply_padding(self, data):
        """
        Apply padding to labels or other data arrays.
        """
        height, width = data.shape
        pad_h = (self.output_size[0] - height % self.output_size[0]) % self.output_size[0]
        pad_w = (self.output_size[1] - width % self.output_size[1]) % self.output_size[1]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        data_padded = F.pad(
            data,
            pad=(pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
            mode="constant",
            value=0
        )

        data_padded = data_padded.squeeze(0).squeeze(0)  # (H_padded, W_padded)

        return data_padded

    def map_mask_to_discrete_classes(self, mask, encoder):
        max_label = max(encoder.keys())

        mapping = torch.zeros(max_label + 1, dtype=torch.long, device=mask.device)

        for original_class, new_class in encoder.items():
            mapping[original_class] = new_class

        mapped_mask = mapping[mask]

        return mapped_mask

    def plot(self, sample, suptitle=None):
        rgb_bands = ["B04", "B03", "B02"]

        if not all(band in self.bands for band in rgb_bands):
            warnings.warn("No RGB image.")  # noqa: B028
            return None

        sample_image = sample["image"]
        rgb_band_indices = [self.bands.index(band) for band in rgb_bands]
        rgb_images = []
        for t in range(sample_image.shape[0]):
            rgb_image = sample_image[t, rgb_band_indices, :, :].transpose(1, 2, 0)

            # Normalization
            rgb_min = rgb_image.min(axis=(0, 1), keepdims=True)
            rgb_max = rgb_image.max(axis=(0, 1), keepdims=True)
            denom = rgb_max - rgb_min
            denom[denom == 0] = 1
            rgb_image = (rgb_image - rgb_min) / denom

            rgb_images.append(np.clip(rgb_image, 0, 1))

        return self._plot_sample(rgb_images, sample.get("labels"), suptitle=suptitle)

    def _plot_sample(self, images, labels=None, suptitle=None):
        num_images = len(images)
        cols = 5
        rows = (num_images + cols) // cols

        fig, ax = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        if rows == 1:
            ax = np.expand_dims(ax, 0)
        if cols == 1:
            ax = np.expand_dims(ax, 1)

        for i, image in enumerate(images):
            ax[i // cols, i % cols].imshow(image)
            ax[i // cols, i % cols].set_title(f"Image {i + 1}")
            ax[i // cols, i % cols].axis("off")

        if labels is not None:
            if rows * cols > num_images:
                target_ax = ax[(num_images) // cols, (num_images) % cols]
            else:
                fig.add_subplot(rows + 1, 1, 1)
                target_ax = fig.gca()

            target_ax.imshow(labels.numpy(), cmap="tab20")
            target_ax.set_title("Labels")
            target_ax.axis("off")

        for k in range(num_images, rows * cols):
            ax[k // cols, k % cols].axis("off")

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        plt.show()
