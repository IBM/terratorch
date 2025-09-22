import argparse
import torch
import torch.nn as nn
from jsonargparse import ArgumentParser
from terratorch.tasks import SemanticSegmentationTask
from torch.onnx import dynamo_export


def load_task_from_config(config_path: str) -> SemanticSegmentationTask:
    parser = ArgumentParser()
    parser.add_class_arguments(SemanticSegmentationTask, "model")
    cfg = parser.parse_path(config_path)
    namespace = parser.instantiate_classes(cfg)
    return namespace.model


class ONNXWrapper(nn.Module):
    """Wraps TerraTorch task model to return only raw tensor output."""

    def __init__(self, task: SemanticSegmentationTask):
        super().__init__()
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.task(x)
        # unwrap ModelOutput (assumes `output` field carries logits/tensor)
        if hasattr(out, "output"):
            return out.output
        elif isinstance(out, (tuple, list)):
            return out[0]
        else:
            return out


def export_to_onnx(model, input_shape, output_path):
    model.eval()
    dummy_input = torch.randn(*input_shape)

    export_output = dynamo_export(model, dummy_input)
    export_output.save(output_path)
    print(f"Exported with dynamo_export to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export TerraTorch model to ONNX")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (yaml/json)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .ckpt")
    parser.add_argument("--input-shape", type=int, nargs="+", required=True,
                        help="Input shape, e.g. --input-shape 1 12 256 256")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX filename")
    args = parser.parse_args()

    # load task from config
    task = load_task_from_config(args.config)
    # restore weights
    task = SemanticSegmentationTask.load_from_checkpoint(args.checkpoint, **task.hparams)

    model = task.model
    model.eval()  # Set the model to evaluation mode

    # Step 3: Create a random input tensor.
    # Your model expects [batch, channels, height, width]
    # The ONNX version was [1, 4, 224, 224]
    random_input = torch.randn(1, 4, 224, 224)

    # Step 4: Run inference.
    with torch.no_grad():
        output = model(random_input)

    # Step 5: Check the output.
    # The output is a tensor of logits [1, 7, 224, 224]
    # Let's find the predicted class for the first few pixels to see if it varies.
    # Use argmax to get the class with the highest score for each pixel
    predicted_classes = torch.argmax(output.output, dim=1)
    print("Predicted classes for random input:")
    print(predicted_classes.flatten()[:10000])
    print((predicted_classes.flatten() == 2).all())
    hist = torch.histc(predicted_classes.flatten().float(), bins=7, min=-10, max=10)
    print(hist)

    from terratorch.datasets.m_chesapeake_landcover import MChesapeakeLandcoverNonGeo
    from torchvision.transforms import CenterCrop, ToTensor

    dataset = MChesapeakeLandcoverNonGeo(
        split="de-test",
        download=True,
        cache="chesapeake"   # local cache folder
    )
    x, y = dataset[0]  # first patch (x = 4 channels, y = mask)

    x = x.unsqueeze(0)  # add batch dim
    model.eval()
    with torch.no_grad():
        out = model(x)

    pred = torch.argmax(out.output, dim=1)

    print(torch.unique(pred))

    # wrap so ONNX export sees only tensor outputs
    #wrapped_model = ONNXWrapper(task.model)

    #export_to_onnx(wrapped_model, tuple(args.input_shape), args.output)


if __name__ == "__main__":
    main()
