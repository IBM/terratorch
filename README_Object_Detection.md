# TerraTorch Object Detection Support

## Quick start

```
$ conda create -n terratorch python=3.11
$ conda activate terratorch
$ git clone -b obj_det_geobench git@github.com:IBM/terratorch.git
$ cd terratorch
$ pip install -r requirements/required.txt -r requirements/dev.txt
$ pip install pycocotools

# Use latest version of Kornia from github since Kornia 0.7.4 (relased latest one) has data movement issue (https://github.com/kornia/kornia/issues/3066)
$ pip uninstall kornia
$ pip install git+https://github.com/kornia/kornia.git

$ pip install -e .

$ cd terratorch/examples/confs
$ terratorch fit --config object_detection_vhr10.yaml
```
