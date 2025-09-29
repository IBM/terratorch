models = {
    "prithvi_300m_sen1floods11": {
        "location": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    },
    "prithvi_300m_burnscars": {
        "location": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
    }
}

inputs = {
    "india_url_in_base64_out":
        {
            "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
            "indices": [1, 2, 3, 8, 11, 12],
            "data_format": "url",
            "out_data_format": "b64_json",
        },
    "valencia_url_in_base64_out":
        {
            "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
            "data_format": "url",
            "out_data_format": "b64_json",
        },
    "valencia_url_in_path_out":
        {
            "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
            "data_format": "url",
            "out_data_format": "path",
        },
    "burnscars_url_in_base64_out":
    {
        "image_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",
        "data_format": "url",
        "out_data_format": "b64_json",

    },
    "burnscars_url_in_path_out":
    {
        "image_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",
        "data_format": "url",
        "out_data_format": "path",
    }
}