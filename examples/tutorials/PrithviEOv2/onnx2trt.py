import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Load ONNX model
with open("model2.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

# Set optimization profile matching 5D input
input_tensor = network.get_input(0)
input_name = input_tensor.name
profile = builder.create_optimization_profile()

# Must be 5D shape: (batch_size, 6, 1, 224, 224)
min_shape = (1, 6, 512, 512)
opt_shape = (1, 6, 512, 512)
max_shape = (1, 6, 512, 512)
profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)

config = builder.create_builder_config()
config.add_optimization_profile(profile)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# Build engine
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build engine")

# Save engine
with open("model.trt", "wb") as f:
    f.write(serialized_engine)
