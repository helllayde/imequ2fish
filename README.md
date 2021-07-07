# IMEQU2FISH

Simple Python Script to convert equirectangular images to double fisheye images using OpenCL for bulk processing.
This script was originally commisioned from a PoliBa university associate.
The script can be used with any hardware configuration capable of running the OpenCL runtime.

## Requirements

This script requires the following packages to run:

* pillow
* pyopencl
* numpy

It may require you to install the Developer Drivers for OpenCL from the official website of [CUDA](https://developer.nvidia.com/cuda-downloads), [AMD](https://developer.amd.com/tools-and-sdks/) or [Intel](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html)

## Usage

In order to use this script, put your equirectangular images in a directory and run the command:

```
py imequ2fish.py -s 'your_source_directory' -d 'your_destination_directory'
```
You may also want to specify the aperture of the image in degrees using the `-a` option or the number of process to start in multiprocessing by using the `-p` option

## License
Copyright (c) 2021 helllayde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
