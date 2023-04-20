% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This hogdraw.m file is part of a Machine Learning project for the university course
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

function hogdraw(hog, cell_size, scale)


% Define the colors for each bin
colors = {'r', 'g', 'b', 'y', 'm', 'c', 'w', 'k', [0.5 0.5 0.5]};

% Compute the dimensions of the HOG features
[num_cells_y, num_cells_x, num_bins] = size(hog);

% Create an empty image to hold the visualization
image = zeros(num_cells_y * cell_size, num_cells_x * cell_size);

% Draw the HOG features for each cell
for y = 1:num_cells_y
  for x = 1:num_cells_x
    % Compute the center of the current cell
    xc = (x - 0.5) * cell_size;
    yc = (y - 0.5) * cell_size;

    % Draw a square to represent the current cell
    x1 = xc - 0.5 * cell_size;
    y1 = yc - 0.5 * cell_size;
    x2 = xc + 0.5 * cell_size;
    y2 = yc + 0.5 * cell_size;
    rectangle('Position', [x1, y1, cell_size, cell_size], 'EdgeColor', 'k', 'LineWidth', 1, 'LineStyle', '-');

    % Draw a line for each bin in the current cell
    for b = 1:num_bins
      % Compute the angle and magnitude of the current bin
      angle = (b - 0.5) * pi / num_bins;
      magnitude = scale * hog(y, x, b);

      % Compute the start and end points of the line for the current bin
      x1 = xc;
      y1 = yc;
      x2 = x1 + magnitude * cos(angle);
      y2 = y1 + magnitude * sin(angle);

      % Draw the line for the current bin
      line([x1 x2], [y1 y2], 'Color', colors{b}, 'LineWidth', 2);
    end
  end
end

% Set the axis limits to show only the image region
axis([0 size(image, 2) 0 size(image, 1)]);
axis equal;
