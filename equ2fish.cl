/*
 * Copyright (C) 2021 helllayde
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#define F_PI 3.14159265358979323846f

/*
 * Kernel to compute the transformation from equirectangular image to fisheye image
 * This Kernel is compilant to OpenCL Standard 1.0
 *
 * Parameters
 * ----------
 *  orig : image2d_t
 *      Original image
 *  fish : image2d_t
 *      Fisheye image bufffer
 *  aperture : float
 *      Aperture in radiants of the photo
 *  width : int
 *      Width of the image in pixels
 *  height : int
 *      Height of the image in pixels
 */
__kernel void equ2fish(read_only const image2d_t orig, write_only image2d_t fish, float aperture, int width, int height)
    {
        const sampler_t sampler = 
            CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

        int2 origPos, fishPos, C;
        float3 P;
        float2 point, origPoint;
        uint4 pix;
        float phi, theta, lon, lat, R;

        //Our starting (r, c) touple on the fisheye image
        fishPos = (int2)(get_global_id(0), get_global_id(1));

        //Calculation of the coordinates of the new reference origin
        C = (int2)(width / 2, height / 2);

        //Calculation of the coordinates in reference to the new origin
        point = (float2) ((float) (fishPos.y - C.x) / C.x, (float)(C.y - fishPos.x) / C.y);
        R = sqrt(point.x * point.x + point.y * point.y);

        //Only if we are in our area of interest
        if(R <= 1)
        {
            //Calculating spherical coordinates
            phi = R * aperture / 2;
            theta = atan2(point.y, point.x);

            //Converting spherical coordinates to a 3D vector
            P.x = R * sin(phi) * cos(theta);
            P.y = R * cos(phi);
            P.z = R * sin(phi) * sin(theta);

            //Converting 3D vector to latitude and longitude
            lon = atan2(P.y, P.x);
            lat = atan2(P.z, sqrt(P.x * P.x + P.y * P.y));

            //Calculating our 2D equirectangular point related from our (r, c) touple
            origPoint = (float2) (lon / F_PI, 2 * lat / F_PI);
            
            //Re-normalization of 2D point to the Image reference origin
            origPos = (int2) (C.x + origPoint.x * C.x, C.y - origPoint.y * C.y);

            //Checking bounduaries
            if(origPos.x < height && origPos.y < width)
            {
                //Trasformation
                pix = read_imageui(orig, sampler, origPos);
                write_imageui(fish, fishPos, pix);
            }
        }
    }