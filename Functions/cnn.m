% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This cnn.m file is part of a Machine Learning project for the university course
% at Laurentian University.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
% --------------------------------------------------------------------------------

% Define the input and output dimensions of the convolutional layer
input_shape = [28, 28, 1];
num_filters = 32;
filter_size = [3, 3];
padding = 'same';
stride = [1, 1];

output_shape = computeOutputSize(input_shape, num_filters, filter_size, padding, stride);

% Initialize the learnable parameters of the layer (the filters and bias terms)
filters = randn([filter_size, input_shape(3), num_filters]);
bias = randn([1, 1, num_filters]);

% Feedforward step
input_data = randn([input_shape, batch_size]);
conv_output = convolve(input_data, filters, bias, padding, stride);
nonlinearity_output = relu(conv_output);
pool_output = maxpool(nonlinearity_output, pool_size, pool_stride);

% Backpropagation step
% Compute the gradient of the loss with respect to the output of the layer
d_pool_output = randn([output_shape, batch_size]);
d_nonlinearity_output = d_pool_output .* kronecker(pool_output);
d_conv_output = d_nonlinearity_output .* relu_grad(conv_output);
d_input_data = deconvolve(d_conv_output, filters, padding, stride);

% Compute the gradient of the loss with respect to the parameters of the layer
d_filters = convolve(input_data, d_conv_output, 'valid');
d_bias = sum(sum(sum(d_conv_output)));

% Update the learnable parameters of the layer using the gradients
learning_rate = 0.001;
filters = filters - learning_rate * d_filters;
bias = bias - learning_rate * d_bias;
