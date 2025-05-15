import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

batch_size = 1
input_shape = (batch_size, 6, 512, 512)
output_shape = (batch_size, 2, 512, 512)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('model.trt', 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_data)

context = engine.create_execution_context()

d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.float32().itemsize)
d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.float32().itemsize)

stream = cuda.Stream()

input_data = np.random.randn(*input_shape).astype(np.float32)

cuda.memcpy_htod(d_input, input_data)

context.execute_v2([int(d_input), int(d_output)])

output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, d_output)

print("Inference completed.")
print("Output shape:", output_data.shape)
