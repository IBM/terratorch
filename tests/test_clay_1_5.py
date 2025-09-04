import gc
import torch
from torchvision.transforms import v2
from terratorch.models.clay1_5_model_factory import Clay1_5ModelFactory



def test_create_model():

    # with open("../model/configs/metadata.yaml", "r") as f:
    #     metadata_contents = yaml.safe_load(f)

    #print(metadata_contents['sentinel-2-l2a']['bands'])

    #print(metadata_contents['sentinel-2-l2a'].bands.wavelength.values())
    
    model_args = {
        # ENCODER
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        # DECODER
        "decoder_dim": 96,
        "decoder_depth": 3,
        "decoder_heads": 2,
        "decoder_dim_head": 48,
        "decoder_mlp_ratio": 2,
        "mask_ratio": 0.75,
        "norm_pix_loss": False,
        "patch_size": 8,
        "shuffle": True,
        "metadata": {
            "sentinel-2-l2a": {
                "band_order": ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"],
                "rgb_indices": [2, 1, 0],
                "gsd": 10,
                "bands": {
                    "mean": {"blue": 1105., "green": 1355., "red": 1552., "rededge1": 1887., "rededge2": 2422., "rededge3": 2630., "nir": 2743., "nir08": 2785., "swir16": 2388., "swir22": 1835.},
                    "std": {"blue": 1809., "green": 1757., "red": 1888., "rededge1": 1870., "rededge2": 1732., "rededge3": 1697., "nir": 1742., "nir08": 1648., "swir16": 1470., "swir22": 1379.},
                    "wavelength": {"blue": 0.493, "green": 0.56, "red": 0.665, "rededge1": 0.704, "rededge2": 0.74, "rededge3": 0.783, "nir": 0.842, "nir08": 0.865, "swir16": 1.61, "swir22": 2.19}
                }
            },
            "planetscope-sr": {
                "band_order": ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"],
                "rgb_indices": [5, 3, 1],
                "gsd": 5,
                "bands": {
                    "mean": {"coastal_blue": 1720., "blue": 1715., "green_i": 1913., "green": 2088., "yellow": 2274., "red": 2290., "rededge": 2613., "nir": 3970.},
                    "std": {"coastal_blue": 747., "blue": 698., "green_i": 739., "green": 768., "yellow": 849., "red": 868., "rededge": 849., "nir": 914.},
                    "wavelength": {"coastal_blue": 0.443, "blue": 0.490, "green_i": 0.531, "green": 0.565, "yellow": 0.610, "red": 0.665, "rededge": 0.705, "nir": 0.865}
                }
            },
            "landsat-c2l1": {
                "band_order": ["red", "green", "blue", "nir08", "swir16", "swir22"],
                "rgb_indices": [0, 1, 2],
                "gsd": 30,
                "bands": {
                    "mean": {"red": 10678., "green": 10563., "blue": 11083., "nir08": 14792., "swir16": 12276., "swir22": 10114.},
                    "std": {"red": 6025., "green": 5411., "blue": 5468., "nir08": 6746., "swir16": 5897., "swir22": 4850.},
                    "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir08": 0.86, "swir16": 1.6, "swir22": 2.2}
                }
            },
            "landsat-c2l2-sr": {
                "band_order": ["red", "green", "blue", "nir08", "swir16", "swir22"],
                "rgb_indices": [0, 1, 2],
                "gsd": 30,
                "bands": {
                    "mean": {"red": 13705., "green": 13310., "blue": 12474., "nir08": 17801., "swir16": 14615., "swir22": 12701.},
                    "std": {"red": 9578., "green": 9408., "blue": 10144., "nir08": 8277., "swir16": 5300., "swir22": 4522.},
                    "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir08": 0.86, "swir16": 1.6, "swir22": 2.2}
                }
            },
            "naip": {
                "band_order": ["red", "green", "blue", "nir"],
                "rgb_indices": [0, 1, 2],
                "gsd": 1.0,
                "bands": {
                    "mean": {"red": 110.16, "green": 115.41, "blue": 98.15, "nir": 139.04},
                    "std": {"red": 47.23, "green": 39.82, "blue": 35.43, "nir": 49.86},
                    "wavelength": {"red": 0.65, "green": 0.56, "blue": 0.48, "nir": 0.842}
                }
            },
            "linz": {
                "band_order": ["red", "green", "blue"],
                "rgb_indices": [0, 1, 2],
                "gsd": 0.5,
                "bands": {
                    "mean": {"red": 89.96, "green": 99.46, "blue": 89.51},
                    "std": {"red": 41.83, "green": 36.96, "blue": 31.45},
                    "wavelength": {"red": 0.635, "green": 0.555, "blue": 0.465}
                }
            },
            "sentinel-1-rtc": {
                "band_order": ["vv", "vh"],
                "gsd": 10,
                "bands": {
                    "mean": {"vv": -12.113, "vh": -18.673},
                    "std": {"vv": 8.314, "vh": 8.017},
                    "wavelength": {"vv": 3.5, "vh": 4.0}
                }
            },
            "modis": {
                "band_order": ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"],
                "rgb_indices": [0, 3, 2],
                "gsd": 500,
                "bands": {
                    "mean": {
                        "sur_refl_b01": 1072., "sur_refl_b02": 1624., "sur_refl_b03": 931., "sur_refl_b04": 1023.,
                        "sur_refl_b05": 1599., "sur_refl_b06": 1404., "sur_refl_b07": 1051.
                    },
                    "std": {
                        "sur_refl_b01": 1643., "sur_refl_b02": 1878., "sur_refl_b03": 1449., "sur_refl_b04": 1538.,
                        "sur_refl_b05": 1763., "sur_refl_b06": 1618., "sur_refl_b07": 1396.
                    },
                    "wavelength": {
                        "sur_refl_b01": .645, "sur_refl_b02": .858, "sur_refl_b03": .469, "sur_refl_b04": .555,
                        "sur_refl_b05": 1.240, "sur_refl_b06": 1.640, "sur_refl_b07": 2.130
                    }
                }
            }
        },
        "teacher": "vit_large_patch14_reg4_dinov2.lvd142m",
        "dolls": [16, 32, 64, 128, 256, 768, 1024],
        "doll_weights": [1, 1, 1, 1, 1, 1, 1],
        "in_channels": None,
        "batch_size": 1,
        "platform": ["sentinel-2-l2a"]
    }

    clay_model = Clay1_5ModelFactory().build_model(1, 3, ["sentinel-2-l2a"], **model_args)
    # data_cube = {
    #     "pixels": torch.randn(64, 10, 64, 64), 
    #     "time": torch.stack([torch.zeros(4) for _ in range(64)]),
    #     "platform": ["sentinel-2-l2a"],
    #     "latlon": torch.zeros(64,4),
    #     "waves": torch.zeros(4),
    #     "gsd": torch.tensor(10),
    # }

    # clay_model.forward(data_cube)
    clay_model.forward(torch.randn(1, 10, 224, 224))

    gc.collect()

