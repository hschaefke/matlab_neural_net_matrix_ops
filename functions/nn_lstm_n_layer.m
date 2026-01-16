function [x_k_plus_1, h_k, c_k] = nn_lstm_n_layer(x_k, h_k_minus_1, c_k_minus_1, lstm_weights) %#codegen
% Computes the forward pass of an n-layer LSTM network using only
% matrix operations.
%
% INPUTS:
%   x_k           : Input at time step k           [input_dim, 1]
%   h_k_minus_1   : Previous hidden states          [hidden_dim, num_layers]
%                   Column l corresponds to layer l (0-based index in weights).
%                   (If a vector [hidden_dim, 1] is provided, it will be expanded.)
%   c_k_minus_1   : Previous cell states            [hidden_dim, num_layers]
%                   Column l corresponds to layer l (0-based index in weights).
%                   (If a vector [hidden_dim, 1] is provided, it will be expanded.)
%   lstm_weights  : Struct containing all LSTM parameters with fields:
%
%       For each layer l = 0, 1, ..., num_layers-1:
%
%       Input gate (i):
%           w_ii_l<l> : Input-to-input-gate weights  [hidden_dim, input_dim_l]
%           b_ii_l<l> : Bias                         [hidden_dim, 1]
%           w_hi_l<l> : Hidden-to-input-gate weights [hidden_dim, hidden_dim]
%           b_hi_l<l> : Bias                         [hidden_dim, 1]
%
%       Forget gate (f):
%           w_if_l<l>, b_if_l<l>, w_hf_l<l>, b_hf_l<l>
%
%       Cell candidate (g):
%           w_ig_l<l>, b_ig_l<l>, w_hg_l<l>, b_hg_l<l>
%
%       Output gate (o):
%           w_io_l<l>, b_io_l<l>, w_ho_l<l>, b_ho_l<l>
%
%       Linear output layer:
%           weight_linout : [output_dim, hidden_dim]
%           bias_linout   : [output_dim, 1]
%
% OUTPUTS:
%   x_k_plus_1 : Network output at time k+1        [output_dim, 1]
%   h_k        : Updated hidden states              [hidden_dim, num_layers]
%   c_k        : Updated cell states                [hidden_dim, num_layers]
%
% -------------------------------------------------------------------------

    w = lstm_weights; % Shortcut to improve readability

    % Infer number of layers from the struct:
    % Each LSTM layer contributes 16 fields (4 gates * (W_x, b_x, W_h, b_h)),
    % plus 2 fields for the linear output layer.
    num_layers = (numel(fieldnames(w)) - 2) / 16;
    hidden_dim = size(w.weight_linout, 2);

    % Accept both vector and matrix state inputs
    if isvector(h_k_minus_1)
        h_k_minus_1 = reshape(h_k_minus_1, hidden_dim, 1);
        if num_layers > 1
            h_k_minus_1 = repmat(h_k_minus_1, 1, num_layers);
        end
    end
    if isvector(c_k_minus_1)
        c_k_minus_1 = reshape(c_k_minus_1, hidden_dim, 1);
        if num_layers > 1
            c_k_minus_1 = repmat(c_k_minus_1, 1, num_layers);
        end
    end

    % Preallocate matrices for updated states
    h_k = zeros(hidden_dim, num_layers);
    c_k = zeros(hidden_dim, num_layers);

    % First layer sees x_k; higher layers see previous layer hidden state
    layer_input = x_k;

    %% ----- LSTM Layers -------------------------------------------------
    for layer = 0:(num_layers - 1)

        h_prev = h_k_minus_1(:, layer + 1);  % Previous hidden state of this layer
        c_prev = c_k_minus_1(:, layer + 1);  % Previous cell state of this layer

        % Fetch weights/biases for this layer
        w_ii = w.(['w_ii_l' num2str(layer)]);
        b_ii = w.(['b_ii_l' num2str(layer)]);
        w_hi = w.(['w_hi_l' num2str(layer)]);
        b_hi = w.(['b_hi_l' num2str(layer)]);

        w_if = w.(['w_if_l' num2str(layer)]);
        b_if = w.(['b_if_l' num2str(layer)]);
        w_hf = w.(['w_hf_l' num2str(layer)]);
        b_hf = w.(['b_hf_l' num2str(layer)]);

        w_ig = w.(['w_ig_l' num2str(layer)]);
        b_ig = w.(['b_ig_l' num2str(layer)]);
        w_hg = w.(['w_hg_l' num2str(layer)]);
        b_hg = w.(['b_hg_l' num2str(layer)]);

        w_io = w.(['w_io_l' num2str(layer)]);
        b_io = w.(['b_io_l' num2str(layer)]);
        w_ho = w.(['w_ho_l' num2str(layer)]);
        b_ho = w.(['b_ho_l' num2str(layer)]);

        %% ----- Input Gate (i_k) ----------------------------------------
        i_k = sigmoid( w_ii * layer_input + b_ii + w_hi * h_prev + b_hi );

        %% ----- Forget Gate (f_k) ---------------------------------------
        f_k = sigmoid( w_if * layer_input + b_if + w_hf * h_prev + b_hf );

        %% ----- Cell Candidate (g_k) ------------------------------------
        g_k = tanh(   w_ig * layer_input + b_ig + w_hg * h_prev + b_hg );

        %% ----- Output Gate (o_k) ---------------------------------------
        o_k = sigmoid( w_io * layer_input + b_io + w_ho * h_prev + b_ho );

        %% ----- Cell State Update (c_k) ---------------------------------
        c_layer = f_k .* c_prev + i_k .* g_k;

        %% ----- Hidden State Update (h_k) -------------------------------
        h_layer = o_k .* tanh(c_layer);

        % Store updated states
        h_k(:, layer + 1) = h_layer;
        c_k(:, layer + 1) = c_layer;

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
