# TerraTorch Object Detection Support

## Quick start

```
$ conda create -n terratorch python=3.11
$ conda activate terratorch
$ git clone -b obj_det_geobench git@github.com:IBM/terratorch.git
$ cd terratorch
$ pip install -r requirements/required.txt -r requirements/dev.txt
$ pip install pycocotools  # Needed for ObjectDetectionTask
$ pip install -e .

$ cd terratorch/examples/confs
$ terratorch fit --config object_detection_vhr10.yaml
```
