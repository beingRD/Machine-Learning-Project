% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This split.m file is part of a Machine Learning project for the university course
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

load("CK_data.mat");

% Load the dataset
imshow(data(:,:,:,1));

% Split the data into training, validation, and test sets
train_ratio = 0.6; % 60% of the data will be used for training
val_ratio = 0.2; % 20% of the data will be used for validation
test_ratio = 0.2; % 20% of the data will be used for testing

% Randomly shuffle the data
% Get the number of samples
num_samples = size(data, 4);

% Randomly shuffle the data and labels
idx = randperm(num_samples);
data_shuffled = data(:, :, :, idx);
labels_shuffled = labels(idx);
% Compute the number of samples in each set
num_train_samples = floor(train_ratio * num_samples);
num_val_samples = floor(val_ratio * num_samples);
num_test_samples = num_samples - num_train_samples - num_val_samples;

% Split the data and labels into train, validation, and test sets
train_data = data_shuffled(:, :, :, 1:num_train_samples);
train_labels = labels_shuffled(1:num_train_samples);
val_data = data_shuffled(:, :, :, num_train_samples+1:num_train_samples+num_val_samples);
val_labels = labels_shuffled(num_train_samples+1:num_train_samples+num_val_samples);
test_data = data_shuffled(:, :, :, num_train_samples+num_val_samples+1:end);
test_labels = labels_shuffled(num_train_samples+num_val_samples+1:end);


imshow(test_data(:,:,:,5));
disp(test_labels(5));
