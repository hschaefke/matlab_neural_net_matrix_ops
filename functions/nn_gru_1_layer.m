function [x_k_plus_1, h_k] = nn_gru_1_layer(x_k, h_k_minus_1, gru_weights)
% Computes the forward pass of a single-layer GRU network using only
% matrix operations.
%
% INPUTS:
%   x_k           : Input at time step k  [input_dim, 1]
%   h_k_minus_1   : Previous hidden state [hidden_dim, 1]
%   gru_weights   : Struct containing all GRU parameters with the fields:
%
%       Reset Gate:
%           w_ir_l0   : Input-to-reset weights  [hidden_dim, input_dim]
%           b_ir_l0   : Bias                     [hidden_dim, 1]
%           w_hr_l0   : Hidden-to-reset weights [hidden_dim, hidden_dim]
%           b_hr_l0   : Bias                     [hidden_dim, 1]
%
%       Update Gate:
%           w_iz_l0, b_iz_l0, w_hz_l0, b_hz_l0
%
%       New Gate:
%           w_in_l0, b_in_l0, w_hn_l0, b_hn_l0
%
%       Output Layer:
%           weight_linout : [output_dim, hidden_dim]
%           bias_linout   : [output_dim, 1]
%
% OUTPUTS:
%   x_k_plus_1 : Network output at time k+1     [output_dim, 1]
%   h_k        : Updated hidden state            [hidden_dim, 1]
%
% -------------------------------------------------------------------------

    w = gru_weights; % Shortcut to improve readability
    h_prev = h_k_minus_1;  % Previous hidden state

    %% ----- Reset Gate (r_k) --------------------------------------------
    r_k = sigmoid( w.w_ir_l0 * x_k + w.b_ir_l0 + w.w_hr_l0 * h_prev + w.b_hr_l0 );

    %% ----- Update Gate (z_k) -------------------------------------------
    z_k = sigmoid( w.w_iz_l0 * x_k + w.b_iz_l0 + w.w_hz_l0 * h_prev + w.b_hz_l0 );

    %% ----- New Gate (n_k) ----------------------------------------------
    n_k = tanh( w.w_in_l0 * x_k + w.b_in_l0 + r_k .* ( w.w_hn_l0 * h_prev + w.b_hn_l0 ) );

    %% ----- Hidden State Update -----------------------------------------
    h_k = (1 - z_k) .* n_k + z_k .* h_prev;

    %% ----- Linear Output Layer -----------------------------------------
    x_k_plus_1 = w.weight_linout * h_k + w.bias_linout;

end


%% ======================================================================
%  Sigmoid activation function
% ======================================================================
function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end
