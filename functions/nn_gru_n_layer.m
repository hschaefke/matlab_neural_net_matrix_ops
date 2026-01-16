function [x_k_plus_1, h_k] = nn_gru_n_layer(x_k, h_k_minus_1, gru_weights) %#codegen
% Computes the forward pass of an n-layer GRU network using only
% matrix operations.
%
% INPUTS:
%   x_k           : Input at time step k  [input_dim, 1]
%   h_k_minus_1   : Previous hidden states [hidden_dim, num_layers]
%                   Column l corresponds to layer l (0-based index in weights).
%                   (If a vector [hidden_dim, 1] is provided, it will be expanded.)
%   gru_weights   : Struct containing all GRU parameters with the fields:
%
%       For each layer l = 0, 1, ..., num_layers-1:
%
%       Reset Gate:
%           w_ir_l<l>   : Input-to-reset weights  [hidden_dim, input_dim_l]
%           b_ir_l<l>   : Bias                     [hidden_dim, 1]
%           w_hr_l<l>   : Hidden-to-reset weights [hidden_dim, hidden_dim]
%           b_hr_l<l>   : Bias                     [hidden_dim, 1]
%
%       Update Gate:
%           w_iz_l<l>, b_iz_l<l>, w_hz_l<l>, b_hz_l<l>
%
%       New Gate:
%           w_in_l<l>, b_in_l<l>, w_hn_l<l>, b_hn_l<l>
%
%       Output Layer:
%           weight_linout : [output_dim, hidden_dim]
%           bias_linout   : [output_dim, 1]
%
% OUTPUTS:
%   x_k_plus_1 : Network output at time k+1     [output_dim, 1]
%   h_k        : Updated hidden states           [hidden_dim, num_layers]
%
% -------------------------------------------------------------------------

    w = gru_weights; % Shortcut to improve readability

    % Infer number of layers from the struct:
    % Each GRU layer contributes 12 fields (Reset/Update/New gates),
    % plus 2 fields for the linear output layer.
    num_layers = (numel(fieldnames(w)) - 2) / 12;
    hidden_dim = size(w.weight_linout, 2);

    % Accept both vector and matrix hidden state inputs
    if isvector(h_k_minus_1)
        h_k_minus_1 = reshape(h_k_minus_1, hidden_dim, 1);
        if num_layers > 1
            h_k_minus_1 = repmat(h_k_minus_1, 1, num_layers);
        end
    end

    % Preallocate matrix for updated hidden states
    h_k = zeros(hidden_dim, num_layers);

    % First layer sees x_k; higher layers see previous layer hidden state
    layer_input = x_k;

    %% ----- GRU Layers --------------------------------------------------
    for layer = 0:(num_layers - 1)

        h_prev = h_k_minus_1(:, layer + 1);  % Previous hidden state of this layer

        % Fetch weights/biases for this layer
        w_ir = w.(['w_ir_l' num2str(layer)]);
        b_ir = w.(['b_ir_l' num2str(layer)]);
        w_hr = w.(['w_hr_l' num2str(layer)]);
        b_hr = w.(['b_hr_l' num2str(layer)]);

        w_iz = w.(['w_iz_l' num2str(layer)]);
        b_iz = w.(['b_iz_l' num2str(layer)]);
        w_hz = w.(['w_hz_l' num2str(layer)]);
        b_hz = w.(['b_hz_l' num2str(layer)]);

        w_in = w.(['w_in_l' num2str(layer)]);
        b_in = w.(['b_in_l' num2str(layer)]);
        w_hn = w.(['w_hn_l' num2str(layer)]);
        b_hn = w.(['b_hn_l' num2str(layer)]);

        %% ----- Reset Gate (r_k) ----------------------------------------
        r_k = sigmoid( w_ir * layer_input + b_ir + w_hr * h_prev + b_hr );

        %% ----- Update Gate (z_k) ---------------------------------------
        z_k = sigmoid( w_iz * layer_input + b_iz + w_hz * h_prev + b_hz );

        %% ----- New Gate (n_k) ------------------------------------------
        n_k = tanh( w_in * layer_input + b_in + r_k .* ( w_hn * h_prev + b_hn ) );

        %% ----- Hidden State Update -------------------------------------
        h_layer = (1 - z_k) .* n_k + z_k .* h_prev;

        % Store updated hidden state
        h_k(:, layer + 1) = h_layer;

        % Output of this layer becomes input to the next layer
        layer_input = h_layer;

    end

    %% ----- Linear Output Layer -----------------------------------------
    x_k_plus_1 = w.weight_linout * h_k(:, end) + w.bias_linout;

end


%% ======================================================================
%  Sigmoid activation function
% ======================================================================
function out = sigmoid(x)
    out = 1 ./ (1 + exp(-x));
end
