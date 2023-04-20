# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
# All rights reserved.
#
# This Soft_m.py file is part of a Machine Learning project for the university course
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


class Softmax:

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward_pass(self, input):
        '''
  forward propogation of softmax 
    '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        ttl = np.dot(input, self.weights) + self.biases
        self.last_totals = ttl

        exp_return = np.exp(ttl)
        return exp_return / np.sum(exp_return, axis=0)

    def backprop_pass(self, d_L_d_out, learn_rate):
        '''
        backropogation for softmax layer
    '''
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            total_exponetial = np.exp(self.last_totals)

            # Sum of all e^totals
            Sum = np.sum(total_exponetial)

            Gradient_doutdt = - \
                total_exponetial[i] * total_exponetial / (Sum ** 2)
            Gradient_doutdt[i] = total_exponetial[i] * \
                (Sum - total_exponetial[i]) / (Sum ** 2)

            Gradient_dtdw = self.last_input
            Gradient_dtdb = 1
            Gradient_dtd_inputs = self.weights

            # Gradients of loss against totals
            Gradient_dldt = gradient * Gradient_doutdt

            # Gradients of loss against weights/biases/input
            dldw = Gradient_dtdw[np.newaxis].T @ Gradient_dldt[np.newaxis]
            dldb = Gradient_dldt * Gradient_dtdb
            dld_inputs = Gradient_dtd_inputs @ Gradient_dldt

            # Update weights / biases
            self.weights -= learn_rate * dldw
            self.biases -= learn_rate * dldb

            return dld_inputs.reshape(self.last_input_shape)
