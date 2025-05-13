import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
onnx_model_path = "model.onnx"
trt_engine_path = "model.trt"

builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags=network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_model_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

input_tensor = network.get_input(0)
profile = builder.create_optimization_profile()
profile.set_shape(input_tensor.name,
                  min=(1, 6, 1, 224, 224),
                  opt=(2, 6, 1, 224, 224),
                  max=(4, 6, 1, 224, 224))
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build the engine.")

with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)

print("TensorRT engine successfully created and saved to", trt_engine_path)
