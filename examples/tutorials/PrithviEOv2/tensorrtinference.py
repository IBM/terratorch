import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Define the input shape and output shape
batch_size = 1
input_shape = (batch_size, 6, 512, 512)
output_shape = (batch_size, 2, 512, 512)  # Assuming 2 output classes for segmentation

# Load the serialized TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('model.trt', 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

# Create execution context
context = engine.create_execution_context()

# Allocate memory for inputs and outputs
d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.float32().itemsize)
d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.float32().itemsize)

# Create a stream
stream = cuda.Stream()

# Prepare the input tensor (random data in this case)
input_data = np.random.randn(*input_shape).astype(np.float32)

# Transfer data to the device
cuda.memcpy_htod(d_input, input_data)

# Execute inference
context.execute_v2([int(d_input), int(d_output)])

# Transfer the output data back to the host
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)

# Post-processing the result (optional)
print("Inference completed.")
print("Output shape:", output_data.shape)
