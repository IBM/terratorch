# TerraTorch Object Detection Support

## Set up

```
$ cd WORKSPACE
$ conda create -n terratorch python=3.11
$ conda activate terratorch
$ git clone -b obj_det_geobench git@github.com:IBM/terratorch.git  # Use obj_det_geobench branch
$ cd terratorch
$ pip install -r requirements/required.txt -r requirements/dev.txt
$ pip install pycocotools  # Needed for ObjectDetectionTask
$ pip install -e .
```

Apply a fix provided by Soltius at https://stackoverflow.com/a/72321898/7175848.
Target files are usually found in the directory $HOME/.conda/envs/terratorch-od/lib/python3.11/site-packages/torchvision/models/detection.
This allows Faster-RCNN to return validation loss as well as model output in validation step.
Also model output is returned in training step.

## Execution
```
$ cd terratorch/examples/confs
$ terratorch fit --config object_detection_faster_rcnn_vhr10.yaml
```
