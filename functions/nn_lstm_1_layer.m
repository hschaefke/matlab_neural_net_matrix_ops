function [x_k_plus_1, h_k, c_k] = nn_lstm_1_layer(x_k, h_k_minus_1, c_k_minus_1, lstm_weights)
% Computes the forward pass of a single-layer LSTM network using only
% matrix operations.
%
% INPUTS:
%   x_k           : Input at time step k           [input_dim, 1]
%   h_k_minus_1   : Previous hidden state          [hidden_dim, 1]
%   c_k_minus_1   : Previous cell state            [hidden_dim, 1]
%   lstm_weights  : Struct containing all LSTM parameters with fields:
%
%       Input gate (i):
%           w_ii_l0 : Input-to-input-gate weights  [hidden_dim, input_dim]
%           b_ii_l0 : Bias                         [hidden_dim, 1]
%           w_hi_l0 : Hidden-to-input-gate weights [hidden_dim, hidden_dim]
%           b_hi_l0 : Bias                         [hidden_dim, 1]
%
%       Forget gate (f):
%           w_if_l0, b_if_l0, w_hf_l0, b_hf_l0
%
%       Cell candidate (g):
%           w_ig_l0, b_ig_l0, w_hg_l0, b_hg_l0
%
%       Output gate (o):
%           w_io_l0, b_io_l0, w_ho_l0, b_ho_l0
%
%       Output Layer:
%           weight_linout : [output_dim, hidden_dim]
%           bias_linout   : [output_dim, 1]
%
% OUTPUTS:
%   x_k_plus_1 : Network output at time k+1        [output_dim, 1]
%   h_k        : Updated hidden state              [hidden_dim, 1]
%   c_k        : Updated cell state                [hidden_dim, 1]
%
% -------------------------------------------------------------------------

    w = lstm_weights; % Shortcut to improve readability
    h_prev = h_k_minus_1;  % Previous hidden state
    c_prev = c_k_minus_1;  % Previous cell state

    %% ----- Input Gate (i_k) --------------------------------------------
    i_k = sigmoid( w.w_ii_l0 * x_k + w.b_ii_l0 + w.w_hi_l0 * h_prev + w.b_hi_l0 );

    %% ----- Forget Gate (f_k) -------------------------------------------
    f_k = sigmoid( w.w_if_l0 * x_k + w.b_if_l0 + w.w_hf_l0 * h_prev + w.b_hf_l0 );

    %% ----- Cell Candidate (g_k) ----------------------------------------
    g_k = tanh(    w.w_ig_l0 * x_k + w.b_ig_l0 + w.w_hg_l0 * h_prev + w.b_hg_l0 );

    %% ----- Output Gate (o_k) -------------------------------------------
    o_k = sigmoid( w.w_io_l0 * x_k + w.b_io_l0 + w.w_ho_l0 * h_prev + w.b_ho_l0 );

    %% ----- Cell State Update -------------------------------------------
    c_k = f_k .* c_prev + i_k .* g_k;

    %% ----- Hidden State Update -----------------------------------------
    h_k = o_k .* tanh(c_k);

    %% ----- Linear Output Layer -----------------------------------------
    x_k_plus_1 = w.weight_linout * h_k + w.bias_linout;

end


%% ======================================================================
%  Sigmoid activation function
% ======================================================================
function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end
