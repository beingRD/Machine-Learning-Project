% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This convolce_2.m file is part of a Machine Learning project for the university course
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

function conv_output = convolce_2(X, kernel, num_layers, filter_size)
% X: input data batch, shape = (batch_size, height, width)
% kernel: filter/kernel, shape = (filter_size, filter_size)
% num_layers: number of layers in the input data (e.g., for 2D grayscale image, num_layers = 1)
% filter_size: size of the filter/kernel

[batch_size,~, ~] = size(X);
 [height, width]=size(X{1});
% disp(kernel);
conv_output = zeros(height, width, batch_size);
% disp(filter_size);
for n = 1:batch_size

image = X{n};% Convert to grayscale if necessary
if size(image, 3) == 3
    image = rgb2gray(image);
end
% Resize image to 48x48
image = imresize(image, [48, 48]);

% Define a test kernel


% Set the number of layers and filter size
numLayers = 1;
filterSize = 3;

% Perform convolution on the image using the defined kernel, number of layers, and filter size
convImage = convolve(image, kernel, numLayers, filterSize);
conv_output(:,:,n)=convImage;
end

end
