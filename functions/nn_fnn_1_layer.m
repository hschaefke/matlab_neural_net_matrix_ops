function y_k = nn_fnn_1_layer(x_k, fnn_weights)
% Computes the forward pass of a single-hidden-layer feedforward neural
% network (FNN) using only matrix operations.
%
% INPUTS:
%   x_k         : Input at time step k  [input_dim, 1]
%   fnn_weights : Struct containing all FNN parameters with the fields:
%
%       Hidden Layer:
%           linear_0_weight : [hidden_dim, input_dim]
%           linear_0_bias   : [hidden_dim, 1]
%
%       Output Layer:
%           linear_1_weight : [output_dim, hidden_dim]
%           linear_1_bias   : [output_dim, 1]
%
% OUTPUTS:
%   y_k : Network output at time step k           [output_dim, 1]
%
% -------------------------------------------------------------------------

    w = fnn_weights; % Shortcut to improve readability

    %% ----- Hidden Layer ------------------------------------------------
    % a_0 = W0*x + b0
    a0 = w.linear_0_weight * x_k + w.linear_0_bias;

    % h_0 = tanh(a_0)
    h0 = tanh(a0);

    %% ----- Output Layer ------------------------------------------------
    % y = W1*h + b1
    y_k = w.linear_1_weight * h0 + w.linear_1_bias;

end
