% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This Hog_extract.m file is part of a Machine Learning project for the university course
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

addpath("Functions");
% Define the parameters for HOG computation
cell_size = 8;
num_bins = 9;

% Create an ImageDatastore object to read images from the CK+ dataset folder
imds = imageDatastore('CK+', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Initialize an empty matrix to hold the extracted HOG features
hog_features = [];

% Loop over each image in the ImageDatastore
for i=1:981
    % Read the next image from the datastore
    img = imds.readimage(i);
[height, width, num_channels] = size(img);
num_cells_x = floor(width / cell_size);
num_cells_y = floor(height / cell_size);
    % Compute the HOG features for the current image
    hog =hog_f([8 8],9,img);
    hog_features_reshaped = reshape(hog, [num_cells_y, num_cells_x, num_bins]);

    % Append the HOG features for the current image to the matrix
    hog_features = [hog_features; hog_features_reshaped];
end
save('hog_f.mat',"hog_features");
