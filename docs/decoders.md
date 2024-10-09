# Decoders


## MMSegmentation Decoders

MMSegmentation decoders are available through the BACKBONE_REGISTRY. 

!!! warning

    MMSegmentation currently requires `mmcv==2.1.0`. Pre-built wheels for this only exist for `torch==2.1.0`.
    In order to use mmseg without building from source, you must downgrade your `torch` to this version.
    Install mmseg with:
    ``` sh
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.1.0
    pip install regex ftfy mmsegmentation
    ```

    We provide access to mmsegdecoders as an external source of decoders, but are not directly responsible for the maintainence of that library.
