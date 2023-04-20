% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This DFH_preprocess.m file is part of a Machine Learning project for the university course
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

main_folder = 'Datasets/DFH';
subfolders = {'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt'};
new_folder = 'Datasets/DFH_resized_grayscape';

% Loop through subfolders, read images, resize, convert to RGB, and save to new folder
for i = 1:length(subfolders)
    folder_path = fullfile(main_folder, subfolders{i});
    files = dir(fullfile(folder_path, '*.jpg'));
    for j = 1:length(files)
        img_path = fullfile(folder_path, files(j).name);
        img = imread(img_path);
        img = imresize(img, [48, 48]);
        img_grey = rgb2gray(img);
        [~, name, ext] = fileparts(files(j).name);
        new_filename = [name '_rgb' ext];
        new_filepath = fullfile(new_folder, subfolders{i}, new_filename);
        if ~exist(fullfile(new_folder, subfolders{i}), 'dir')
            mkdir(fullfile(new_folder, subfolders{i}));
        end
        imwrite(img_grey, new_filepath);
    end
end
