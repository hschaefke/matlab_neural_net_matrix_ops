function [x_k_plus_1, h_k] = nn_gru_n_layer(x_k, h_k_minus_1, gru_weights) %#codegen
% nn_gru_n_layer
% -------------------------------------------------------------------------
% Computes the forward pass of an n-layer GRU network using only matrix
% operations. All GRU layers are stacked: the hidden state of layer L is
% used as input to layer L+1 at the same time step.
%
% This implementation is intended for use with GRU weights exported from
% Python (e.g. PyTorch, TensorFlow) and is toolbox-free, making it suitable
% for nonlinear MPC, embedded deployment, or detailed inspection of
% recurrent dynamics.
%
% INPUTS:
%   x_k           : Input at time step k                  [input_dim, 1]
%   h_k_minus_1   : Previous hidden states of ALL layers  [hidden_dim, num_layers]
%                   Column j corresponds to layer j-1 (0-based index in weights).
%   gru_weights   : Struct containing all GRU parameters and linear output:
%
%       For each GRU layer l = 0, 1, ..., num_layers-1:
%
%         Reset gate (r):
%           w_ir_l<l> : Input-to-reset weights   [hidden_dim, input_dim_l]
%           b_ir_l<l> : Bias                     [hidden_dim, 1]
%           w_hr_l<l> : Hidden-to-reset weights  [hidden_dim, hidden_dim]
%           b_hr_l<l> : Bias                     [hidden_dim, 1]
%
%         Update gate (z):
%           w_iz_l<l>, b_iz_l<l>, w_hz_l<l>, b_hz_l<l>
%
%         New gate (n):
%           w_in_l<l>, b_in_l<l>, w_hn_l<l>, b_hn_l<l>
%
%       Linear output layer (applied to last GRU layer):
%           weight_linout : [output_dim, hidden_dim]
%           bias_linout   : [output_dim, 1]
%
%       Note: input_dim_l is input_dim for layer 0 and hidden_dim for all
%             subsequent layers (since they receive the previous layer's h).
%
% OUTPUTS:
%   x_k_plus_1 : Network output at time k+1               [output_dim, 1]
%   h_k        : Updated hidden states of ALL layers       [hidden_dim, num_layers]
%
% -------------------------------------------------------------------------

    w = gru_weights;  % Shortcut for readability

    % Infer number of layers from the struct:
    % Each GRU layer contributes 12 fields (4 gates * (W_in, b_in, W_h, b_h)),
    % plus 2 fields for the linear output layer.
    num_layers = (numel(fieldnames(gru_weights)) - 2) / 12;
    hidden_dim = size(gru_weights.weight_linout, 2);

    % Preallocate matrix for new hidden states
    h_k = zeros(hidden_dim, num_layers);

    % Current input to the "stack" – first layer sees x_k,
    % higher layers see the hidden state of the previous layer.
    layer_input = x_k;

    %% ----- Iterate over all GRU layers ---------------------------------
    for layer = 0:(num_layers - 1)
        % Previous hidden state for this layer (column layer+1 in h_k_minus_1)
        h_prev_layer = h_k_minus_1(:, layer + 1);

        % Build dynamic field names for this layer
        % Example: 'w_ir_l0', 'b_ir_l0', etc.
        w_ir = w.(['w_ir_l' num2str(layer)]);
        b_ir = w.(['b_ir_l' num2str(layer)])';
        w_hr = w.(['w_hr_l' num2str(layer)]);
        b_hr = w.(['b_hr_l' num2str(layer)])';

        w_iz = w.(['w_iz_l' num2str(layer)]);
        b_iz = w.(['b_iz_l' num2str(layer)])';
        w_hz = w.(['w_hz_l' num2str(layer)]);
        b_hz = w.(['b_hz_l' num2str(layer)])';

        w_in = w.(['w_in_l' num2str(layer)]);
        b_in = w.(['b_in_l' num2str(layer)])';
        w_hn = w.(['w_hn_l' num2str(layer)]);
        b_hn = w.(['b_hn_l' num2str(layer)])';

        %% ----- Reset Gate r_k (layer) -----------------------------------
        r_k = sigmoid( w_ir * layer_input + b_ir + ...
                       w_hr * h_prev_layer + b_hr );

        %% ----- Update Gate z_k (layer) ----------------------------------
        z_k = sigmoid( w_iz * layer_input + b_iz + ...
                       w_hz * h_prev_layer + b_hz );

        %% ----- New Gate n_k (layer) -------------------------------------
        n_k = tanh( w_in * layer_input + b_in + ...
                    r_k .* (w_hn * h_prev_layer + b_hn) );

        %% ----- Hidden State Update for this layer -----------------------
        h_layer = (1 - z_k) .* n_k + z_k .* h_prev_layer;

        % Store updated hidden state
        h_k(:, layer + 1) = h_layer;

        % Output of this layer becomes input to the next layer
        layer_input = h_layer;
    end

    %% ----- Linear Output Layer (acts on last GRU layer) -----------------
    x_k_plus_1 = w.weight_linout * h_k(:, end) + w.bias_linout';

end


%% ======================================================================
%  Sigmoid activation function
% ======================================================================
function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end
