# tensorrt_l2norm_helper

TensorRT plugin that addresses issue with two unsupported oprations within l2_normalize TensorFlow operation.

```
l2_normalize/Maximum: Unsupported binary op max with constant right
l2_normalize/Rsqrt: Unary not supported for other non-constant node
```

NOTE: [as per NVIDIA](https://devtalk.nvidia.com/default/topic/1043541/tensorrt/tf-sqrt-error-unary-not-supported-for-other-non-constant-node-solved-/post/5294189/#5294189) (r)sqrt operation should be fixed in next release (was not fixed in TensorRT 5.0.6).

# Usage

Prerequisites:
- TensorFlow freezed (and optimized) `.pb` graph.
- CUDA and TensorRT (tested with Jetpack 4.2 on Xavier)
- TensorFlow (tested with 1.13.1)
- cmake >= 3.8

Steps:
- Clone this repo to Nvidia device and put your `.pb` file as `sample.pb`.
- Edit `USER DEFINED VALUES` values in `step01_pb_to_uff.py` and `step02_uff_to_plan.cpp`.

- Compile plugin and scripts
```
mkdir build
cd build
cmake ..
make
cd -
```

- Convert `.pb` to UFF `./step01_pb_to_uff.py`. This should produce `sample.uff`
- Convert UFF to TRT plan `./build/step02_uff_to_plan`. This should produce `sample.engine`

# Example network

This plugin was tested with following [Keras based network](https://github.com/cersar/3D_detection) with [patch](https://gist.github.com/r7vme/4eaf168a18a918c17eee7ddc42629537).

# Notes

- FP16 (half-precision) still not supported (TODO)
- `op_type`'s: 0 - Maximum, 1 - Rsqrt, 2 - Sqrt
