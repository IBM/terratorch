models = {
    "prithvi_300m_sen1floods11": {
        "location": "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
    }
}

inputs = {
    "india_url_in_base64_out":
        {
            "plugin": "terratorch_segmentation",
            "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
            "indices": [1, 2, 3, 8, 11, 12],
            "data_format": "url",
            "out_data_format": "b64_json",
        },
    "valencia_url_in_base64_out":
        {
            "plugin": "terratorch_segmentation",
            "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
            "data_format": "url",
            "out_data_format": "b64_json",
        },
    "valencia_url_in_path_out":
        {
            "plugin": "terratorch_segmentation",
            "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
            "data_format": "url",
            "out_data_format": "path",
        },
}