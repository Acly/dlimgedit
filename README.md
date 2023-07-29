# \<dlimgedit\>

*dlimgedit* is a C++ library for image painting and editing workflows which make use of deep learning.

* Simple high-level C++ API
* Flexible integration (supports dynamic loading via C interface)
* Optimized for minimal copying and overhead
* Fully C++ based neural network inference (via [onnxruntime](https://onnxruntime.ai/))
* *Platforms:* Windows, Linux
* *Backends:* CPU, GPU via DirectML (Windows only), GPU via CUDA (Linux/NVIDIA only)

## Features

### Segmentation

Identify objects in an image and generate masks for them (based on [SegmentAnything](https://segment-anything.com))

```cpp
// Load an image...
Image image = Image::load("example.png");
// ...or use existing image data:
ImageView image(pixel_data, {width, height}, Channels::rgba);

// Analyse the image
Environment env;
Segmentation segmentation = Segmentation::process(image, env);

// Query mask for the object at a certain point in the image:
Image mask = segmentation.compute_mask(Point{220, 355});

// Query mask for the largest object contained in a certain region:
Image mask = segmentation.compute_mask(Region(Point{140, 200}, Extent{300, 300}));
```

Performance is interactive: roughly 500ms for `Segmentation::process` and 80ms per mask on CPU. Running on GPU can be much faster: 50ms and 12ms respectively on RTX4070, with around 500MB of VRAM used.


## Building

Building only requires CMake and a compiler with C++20 support (eg. MSVC 2022 on Windows, GCC 13 on Linux).

Clone the repository:
```sh
git clone https://github.com/Acly/dlimgedit.git
cd dlimgedit
```
Configure:
```sh
mkdir build
cd build
cmake ..
```
Build:
```sh
cmake --build . --config Release
```


## Documentation

The library can be added to existing CMake projects either via `add_subdirectory(dlimgedit/src)` to build from source, or by adding the target from installed binaries with `find_package(dlimgedit)`. Packages should work out of the box on CPU. The `onnxruntime` shared library is installed as a required runtime dependency. Execution on GPU may require further libraries at runtime, see below.

The public API is C++14 compatible.

See the [public header](src/include/dlimgedit/dlimgedit.hpp) for API documentation.

### GPU on Windows (DirectML)

Using `Backend::gpu` on Windows makes use of [DirectML](https://github.com/microsoft/DirectML) to run inference on GPU. A large range of GPUs is supported. Deploying `DirectML.dll` ([nuget](https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.12.0)) next to applications is recommended, otherwise the version of the DLL which ships with Windows will be used, and it is usually too old.

### GPU on Linux (CUDA)

Using `Backend::gpu` on Linux makes use of CUDA to run inference on GPU. This requires the following additional libraries to be installed:
* [NVIDIA CUDA Toolkit (Version 11.x)](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* [NVIDIA cuDNN (Version 8.x for CUDA 11.x)](https://developer.nvidia.com/cudnn)

Refer to [NVIDIA's installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) for detailed instructions.
