% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This convolve.m file is part of a Machine Learning project for the university course
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

function convImage = convolve(image, kernel, numLayers, filterSize)
    % Add padding of 1 to height and width
    image = padarray(image, [1 1], 'symmetric');
    [rows, columns] = size(image);
    
    % Initialize output image to the same size as the original image
    convImage = zeros(rows - 2, columns - 2);
    
    % Loop over interior pixels
    for col = numLayers + 2 : columns - numLayers - 1
        for row = numLayers + 2 : rows - numLayers - 1
            % Convolve kernel with image window
            localSum = 0;
            for c = 1 : filterSize
                ic = col + c - numLayers - 1;
                for r = 1 : filterSize
                    ir = row + r - numLayers - 1;
                    localSum = localSum + double(image(ir, ic)) * kernel(r, c);
                end
            end
            % Assign filtered value to output image
            convImage(row - 1, col - 1) = localSum;
        end
    end
end
