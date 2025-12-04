function [x_k_plus_1, h_k] = nn_fnn_1_layer(x_k, fnn_weights) %#codegen
% Computes the forward pass of a simple 1-layer feedforward neural network
% (fully-connected layer + activation + linear output layer) using only
% matrix operations. This implementation is intended for use with trained
% weights exported from Python and is toolbox-free, making it suitable for
% nonlinear MPC, embedded deployment, or low-level analysis.
%
% INPUTS:
%   x_k         : Input vector at time step k        [input_dim, 1]
%   fnn_weights : Struct containing all FNN parameters with fields:
%
%       Hidden layer:
%           w_hidden : Weights [hidden_dim, input_dim]
%           b_hidden : Bias    [hidden_dim, 1]
%
%       Linear output layer:
%           weight_linout : Weights [output_dim, hidden_dim]
%           bias_linout   : Bias    [output_dim, 1]
%
% OUTPUTS:
%   x_k_plus_1 : Network output                      [output_dim, 1]
%   h_k        : Hidden layer activation             [hidden_dim, 1]
%
% -------------------------------------------------------------------------

    w = fnn_weights;  % Shortcut for readability

    %% ----- Hidden Layer -------------------------------------------------
    % Here we use tanh as activation function.
    % You can switch to ReLU or another activation if required.
    %
    %   h_k = tanh(W_hidden * x_k + b_hidden)
    h_k = tanh( w.w_hidden * x_k + w.b_hidden' );

    %% ----- Linear Output Layer -----------------------------------------
    % Simple affine transformation of the hidden activation:
    %
    %   x_k_plus_1 = W_out * h_k + b_out
    x_k_plus_1 = w.weight_linout * h_k + w.bias_linout';

end
