# Copyright contributors to the Terratorch project

"""Command-line interface to TerraTorch."""

from terratorch.cli_tools import build_lightning_cli
import sys
from huggingface_hub import hf_hub_download
import logging

try:
    from benchmark.main import main as iterate_main

    TERRATORCH_ITERATE_INSTALLED = True

except ImportError:
    TERRATORCH_ITERATE_INSTALLED = False


def main():
    if len(sys.argv) == 1:
        print('usage: terratorch [-h] [-c CONFIG] [--print_config[=flags]] {fit,validate,test,predict,compute_statistics,init,iterate} ...')
        exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "iterate":
        # if user runs "terratorch iterate" and terratorch-iterate has not been installed
        if not TERRATORCH_ITERATE_INSTALLED:
            print(
                (
                    "Error! terratorch-iterate has not been installed. If you want to install it,"
                    "run 'pip install terratorch-iterate'"
                )
            )
        # if user runs "terratorch iterate" and terratorch-iterate has been installed
        else:
            # delete iterate argument
            del sys.argv[1]
            iterate_main()
    elif sys.argv[1] == "init":
        logger = logging.getLogger("terratorch-init")
        logger.info("Initializing TerraTorch...")
        logger.info("Downloading model congfigs...")
        files_to_download = [
            {
                'repo_id': 'ibm-granite/granite-geospatial-biomass',
            },
            {
                'repo_id': 'ibm-granite/granite-geospatial-wxc-downscaling',
            },
            {
                'repo_id': 'ibm-granite/granite-geospatial-canopyheight',
            },
            {
                'repo_id': 'ibm-granite/granite-geospatial-land-surface-temperature',
            },
            {
                'repo_id': 'ibm-granite/granite-geospatial-uki',
            },
            {
                'repo_id': 'ibm-granite/granite-geospatial-uki-flooddetection',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M-rollout',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-WxC-1.0-2300m-gravity-wave-parameterization',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL',
                'filename': 'config.json',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL',
                'filename': 'config.json',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars',
                'filename': 'burn_scars_config.yaml',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M',
                'filename': 'config.json',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11',
            },
            {
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11',
            },
        ]

        for config in files_to_download:
            file_path = hf_hub_download(
                repo_id=config['repo_id'],
                filename=config.get('filename', '') or 'config.yaml',
            )
            logger.info(f"File downloaded to: {file_path}")
    else:
        _ = build_lightning_cli()


if __name__ == "__main__":
    main()
