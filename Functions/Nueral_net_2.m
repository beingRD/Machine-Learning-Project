% --------------------------------------------------------------------------------
% Copyright (c) 2023, Rishabh Dev & Hitesh Chauhan
% All rights reserved.
%
% This Neural_net_2.m file is part of a Machine Learning project for the university course
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

function conv_output = Nueral_net_2(X, y, params, hyperparams, mode)
    % X: input data, shape = (batch_size, height, width, channels)
    % y: true labels, shape = (batch_size, num_classes)
    % params: model parameters (weights and biases)
    % hyperparams: hyperparameters (learning rate, number of filters, etc.)
    % mode: string specifying whether to perform 'train' or 'test' mode

    % Unpack hyperparameters
    num_filters = hyperparams.num_filters;
    filter_size = hyperparams.filter_size;
    stride = hyperparams.stride;
    pool_size = hyperparams.pool_size;
    learning_rate = hyperparams.learning_rate;

    % Unpack parameters
    W = params.W;
    b = params.b;

    % Forward pass
    conv_output = zeros(48, 48, num_filters, numel(X));
    for i = 1:num_filters
        kernel = W(:,:,1,i);
        conv_output(:,:,i,:) = convolce_2(X, kernel, 1, filter_size);
    end
end
