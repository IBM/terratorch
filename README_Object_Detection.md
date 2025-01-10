# TerraTorch Object Detection Support

## Quick start

```
$ conda create -n terratorch python=3.11
$ conda activate terratorch
$ git clone -b obj_det_geobench git@github.com:IBM/terratorch.git
$ cd terratorch
$ pip install -r requirements/required.txt -r requirements/dev.txt
$ pip install pycocotools
$ pip install -e .
# Use latest version of Kornia from github since Kornia 0.7.4 (relased latest one) has data movement issue (https://github.com/kornia/kornia/issues/3066)
$ pip uninstall kornia
$ cd ..
$ git clone git@github.com:kornia/kornia.git
$ pip install -e kornia

$ cd ../terratorch/examples/confs
$ terratorch fit --config object_detection_vhr10.yaml
```
