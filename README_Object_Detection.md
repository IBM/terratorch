# TerraTorch Object Detection Support

## Set up

```
$ cd WORKSPACE
$ conda create -n terratorch python=3.11
$ conda activate terratorch
$ git clone -b obj_det_geobench git@github.com:IBM/terratorch.git
$ cd terratorch
$ pip install -r requirements/required.txt -r requirements/dev.txt
$ pip install pycocotools  # Needed for ObjectDetectionTask
$ pip install -e .
```

Object detection task requires a patch to Faster-RCNN discussed in https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn/65347721#65347721.
It allows Faster-RCNN to return validation loss.
You can use patched version of torchvision as follows.
```
$ pip uninstall torchvision
$ cd WORKSPACE
$ git checkout -b v0.19.1-fasterrcnn git@github.com:takaomoriyama/vision.git
$ pip install -e vision
```

## Execution
```
$ cd WORKSPACE/terratorch/examples/confs
$ terratorch fit --config object_detection_vhr10.yaml
```
