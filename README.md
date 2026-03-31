# CUDA Image Sharpen

A CUDA C++ project for image sharpening on the GPU.

This example is useful for learning how image enhancement filters can be accelerated using parallel GPU kernels.

## Features

- image sharpen filter with CUDA
- C++ and GPU implementation
- basic convolution-style image enhancement
- useful for CUDA learning and practice

## Tech Stack

- C++
- CUDA
- OpenCV

## Project Goal

Through this project, you can learn:

- how sharpening filters work
- how CUDA handles image enhancement in parallel
- how to organize GPU image processing code cleanly

## Future Improvements

- stronger filter variants
- runtime parameter tuning
- stream-based execution
- embedded GPU support

## Related Topics

CUDA, Image Sharpening, GPU Filtering, Image Enhancement, C++, OpenCV

## Author

Harry12345123

## Requirements

Before building this project, make sure your system has:

- CUDA Toolkit
- OpenCV
- CMake 3.18 or later
- C++17 compatible compiler

## Build

Use the following commands to compile the project:

```bash
mkdir build
cd build
cmake ..
make -j

Run

After building, run the program with:

./cuda_image_sharpen input.jpg

Notes
Make sure input.jpg exists in the project root directory.
You can replace the input file with your own image.
The executable name depends on your CMake project configuration.



