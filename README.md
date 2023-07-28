# \<dlimgedit\>

*dlimgedit* is a C++ library for image painting and editing workflows which make use of deep learning.

* Simple high-level C++ API
* Flexible integration (supports dynamic loading via C interface)
* Optimized for minimal copying and overhead
* Fully C++ based neural network inference (via [onnxruntime](https://onnxruntime.ai/))
* *Platforms:* Windows
* *Backends:* CPU, CUDA

## Features

### Segmentation

Identify objects in an image and mask them (based on [SegmentAnything](https://segment-anything.com))

```
// Load an image...
Image image = Image::load("example.png");
// ...or use existing image data:
// ImageView image(pixel_data, {width, height}, Channels::rgba);

// Analyse the image
Environment env;
Segmentation segmentation = Segmentation::process(image, env);

// Query mask for the object at a certain point in the image:
Image mask = segmentation.compute_mask(Point{220, 355});

// Query mask for the largest object contained in a certain region:
Image mask = segmentation.compute_mask(Region(Point{140, 200}, Extent{300, 300}));
```


## Building

Building only requires CMake and a compiler with C++20 support (MSVC 2022 on Windows).

Clone the repository:
```
git clone https://github.com/Acly/dlimgedit.git
cd dlimgedit
```
Configure:
```
mkdir build
cd build
cmake ..
```
Build:
```
cmake --build . --config Release
```


## Using

The library can be added to existing CMake projects either via `add_subdirectory(dlimgedit/src)` to build from source, or by adding the target from pre-built binaries with `find_package`.

See the [public header](src/include/dlimgedit/dlimgedit.hpp) for API documentation.
