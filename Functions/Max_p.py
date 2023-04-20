# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
# All rights reserved.
#
# This Max_p.py file is part of a Machine Learning project for the university course
# at Laurentian University.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------------------

import os
import numpy as np
import imageio


class MaxPool_2X2:
    # A Max Pooling layer using a pool size of 2.

    def iterate_regions(self, image):

        height, width, _ = image.shape
        new_height = height // 2
        new_w = width // 2

        for i in range(new_height):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward_pass(self, input):
        '''
    Performs forward pass for max pool 
    '''
        self.last_input = input

        hieght, width, num_filters = input.shape
        output = np.zeros((hieght // 2, width // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop_pass(self, gradient_dld_out):
        '''
   Does backpropogation for max pool layer and returns the gradient of Error function
     '''
        gradient_dld_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            height, width, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(height):
                for j2 in range(width):
                    for f2 in range(f):

                        if im_region[i2, j2, f2] == amax[f2]:
                            gradient_dld_input[i * 2 + i2, j * 2 +
                                               j2, f2] = gradient_dld_out[i, j, f2]

        return gradient_dld_input
