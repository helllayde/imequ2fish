#!/usr/bin/python

# Copyright (C) 2021 helllayde
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import math as m
import numpy as np
from os import walk
from PIL import Image
import pyopencl as cl
from multiprocessing import Pool
from multiprocessing import freeze_support
import os
import optparse

#Parsing line argument options

parser = optparse.OptionParser()

parser.add_option(
    '-s', 
    '--src',
    action = "store", 
    dest = "source",
    help = "Source directory of the images to elaborate", 
    default = "imgs"
)

parser.add_option(
    '-d', 
    '--dest',
    action = "store", 
    dest = "dest",
    help = "Destination directory of the elaborated images", 
    default = "elaborated"
)

parser.add_option(
    '-a', 
    '--apt',
    action = "store", 
    dest = "aperture",
    help = "Aperture (in degrees) of the images", 
    default = "180"
)

parser.add_option(
    '-p', 
    '--prc',
    action = "store", 
    dest = "processors",
    help = "Numbers of processors to dedicate to the process", 
    default = "8"
)

options, args = parser.parse_args()

#Initializing the OpenCL Environment
platforms = cl.get_platforms()
gpuDevices = platforms[0].get_devices(device_type = cl.device_type.GPU)
ctx = cl.Context(devices = gpuDevices)
source = open('equ2fish.cl', 'r')
prg = cl.Program(ctx, source.read()).build()
mf = cl.mem_flags
kernel = prg.equ2fish

def equ2fish(img, aperture):
    """
        Trasforms a given image to a spherical reference and makes it fisheye 
        
        This funcion will use a OpenCL to compute the trasformation 

        Parameters
        ----------
        img : PIL.Image
            The image to transform
        aperture : float
            Aperture (in radiants of the photo - i.e.: math.pi)
    """
    
    imgarr = np.array(img, dtype = np.uint8)
    fishImg = np.zeros(imgarr.shape, dtype = np.uint8)

    w, h = imgarr.shape[1], imgarr.shape[0]
    
    #Creating GPU memory buffers
    origBuff = cl.image_from_array(ctx, imgarr, 4)
    fishBuff = cl.Image(
        ctx, 
        cl.mem_flags.WRITE_ONLY, 
        cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8), 
        shape = (w, h)
    )
    
    #Creating queue for current process
    queue = cl.CommandQueue(ctx)

    #Launching kernel
    kernel(
        queue, 
        imgarr.shape, 
        None, 
        origBuff, 
        fishBuff, 
        np.float32(aperture), 
        np.int32(imgarr.shape[1]), 
        np.int32(imgarr.shape[0])
    )

    cl.enqueue_copy(queue, fishImg, fishBuff, origin=(0, 0), region=(w, h))

    return Image.fromarray(fishImg).rotate(-90)
          
def runMultiprocessing(func, files, processors):
    """
        Maps a given function on the given list computing each call on a different
        multiprocessor core
        
        Parameters
        ----------
        func
            Function to map
        files : list
            List to map
        processors : int
            Number of processors to use
    """
    with Pool(processes = processors) as pool:
        pool.map(func, files)

def elaborateImage(file):
    """
        Splits the image into the raw images taken by the sensor camera
        
        Parameters
        ----------
        file : str
            File name of the image to open
    """
    raw = Image.open('{0}/{1}'.format(options.source, file)).convert('RGBA')

    #Getting all the parts of the image
    sx1 = raw.crop((0, 0, raw.width / 4, raw.height))
    sx2 = raw.crop((3 * raw.width / 4, 0, raw.width, raw.height))
    dx = raw.crop((raw.width / 4, 0, 3 * raw.width / 4, raw.height))

    #Building up the "left" sensor image
    sx = Image.new('RGBA', (sx1.width + sx2.width, sx1.height))
    sx.paste(sx2, (0, 0))
    sx.paste(sx1, (sx2.width, 0))

    aperture = float(options.aperture) * m.pi / 180

    #Computing fisheye
    fish1 = equ2fish(dx, aperture)
    fish2 = equ2fish(sx, aperture)

    # fish1.show()
    # fish2.show()
    fish1.save('{0}/{1}_1.png'.format(options.dest, os.path.splitext(file)[0]))
    fish2.save('{0}/{1}_2.png'.format(options.dest, os.path.splitext(file)[0]))

def main():
    if not os.path.isdir(options.source):
        print('Cannot find the source directory')
        return

    if not os.path.isdir(options.dest):
        os.mkdir(options.dest)

    filenames = next(walk(options.source), (None, None, []))[2]
    if len(filenames) == 0:
        return

    processors = int(options.processors)

    runMultiprocessing(elaborateImage, filenames, processors)

if __name__ == "__main__":
    freeze_support()
    main()