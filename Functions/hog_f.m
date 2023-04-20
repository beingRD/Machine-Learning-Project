% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This hog_f.m file is part of a Machine Learning project for the university course
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


function features_ret = hog_f(cellsize,numbins,I)
% Define HOG parameters
    cellSize =cellsize;
    numBins = numbins;
    [M, N] = size(I);
    numCells = floor([M N] ./ cellSize);
    numBlocks = numCells;
    hogFeatureSize = numBlocks(1) * numBlocks(2) * numBins;
    
    % Compute the gradients using Sobel filters
    Gx = imfilter(double(I), [-1 0 1], 'symmetric');
    Gy = imfilter(double(I), [-1 0 1]', 'symmetric');
    
    % Compute the gradient magnitudes and orientations
    mag = sqrt(Gx.^2 + Gy.^2);
    theta = atan2(Gy, Gx);
    theta(theta < 0) = theta(theta < 0) + pi;
    
    % Compute the histogram of oriented gradients
    hog = zeros(numBlocks(1), numBlocks(2), numBins);
    
    for i = 1:numBins
        bin = ((i - 1) * pi/numBins < theta) & (theta <= i * pi/numBins);
        for j = 1:numBlocks(1)
            for k = 1:numBlocks(2)
                cell = mag((j-1)*cellSize(1)+1:j*cellSize(1), (k-1)*cellSize(2)+1:k*cellSize(2));
                hog(j,k,i) = sum(cell(bin((j-1)*cellSize(1)+1:j*cellSize(1), (k-1)*cellSize(2)+1:k*cellSize(2))));
            end
        end
    end
hog_features = reshape(hog, 1, hogFeatureSize);
% %normalize function on Hog_features
% epsilon = 0.1;
% for i = 1:size(hog_features,1)-1
%     for j = 1:size(hog_features,2)-1
%         block_norm = norm(squeeze(hog_features(i,j,:))) + norm(squeeze(hog_features(i,j+1,:))) + ...
%                      norm(squeeze(hog_features(i+1,j,:))) + norm(squeeze(hog_features(i+1,j+1,:))) + epsilon;
%         hog_features(i,j,:) = hog_features(i,j,:) / block_norm;
%         hog_features(i,j+1,:) = hog_features(i,j+1,:) / block_norm;
%         hog_features(i+1,j,:) = hog_features(i+1,j,:) / block_norm;
%         hog_features(i+1,j+1,:) = hog_features(i+1,j+1,:) / block_norm;
%     end
% end

%L2 Normalization
hog_features=hog_features(:)/norm(hog_features(:));
features_ret=hog_features;
end
