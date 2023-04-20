% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This Load_Preprocess_save.m file is part of a Machine Learning project for the university course
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

ckplus_data = imageDatastore('Datasets/CK+', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
DFH_data = imageDatastore('Datasets/DFH', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


% all_data = imageDatastore(cat(1, ckplus_data.Files, DFH_data.Files), 'LabelSource', 'foldernames');
ckplus_data = shuffle(ckplus_data);
%making split of [75% 15% 15%] = [Training validation testing]
[train_data, valtest_data] = splitEachLabel(ckplus_data, 0.7, 'randomized');
[val_data, test_data] = splitEachLabel(valtest_data, 0.5, 'randomized');

% Display the number of images in each set
numTrain = numel(train_data.Labels);
numVal = numel(val_data.Labels);
numTest = numel(test_data.Labels);
save('Data/ck_train_data.mat', 'train_data');
save('Data/ck_val_data.mat', 'val_data');
save('Data/ck_test_data.mat', 'test_data');
