function y_k = nn_fnn_n_layer(x_k, fnn_weights)
% Computes the forward pass of an n-layer feedforward neural
% network (FNN) using only matrix operations.
%
% INPUTS:
%   x_k         : Input at time step k  [input_dim, 1]
%   fnn_weights : Struct containing all FNN parameters with the fields:
%
%       For each layer l = 0, 1, ..., L:
%           linear_<l>_weight : [out_dim_l, in_dim_l]
%           linear_<l>_bias   : [out_dim_l, 1]
%
%       Note: Additional fields like
%           linear_<l>_in_features, linear_<l>_out_features
%       are ignored by this function.
%
% OUTPUTS:
%   y_k : Network output at time step k           [output_dim, 1]
%
% -------------------------------------------------------------------------

    w = fnn_weights; % Shortcut to improve readability

    % Detect available linear layer indices from "linear_<i>_weight" fields
    fns = fieldnames(w);
    idx = [];
    for i = 1:numel(fns)
        tok = regexp(fns{i}, '^linear_(\d+)_weight$', 'tokens', 'once');
        if ~isempty(tok)
            idx(end+1) = str2double(tok{1}); %#ok<AGROW>
        end
    end
    if isempty(idx)
        error("No fields matching 'linear_<i>_weight' found in fnn_weights.");
    end

    % Sort and validate indices (must be contiguous: 0,1,2,...,L)
    idx = sort(unique(idx));
    if idx(1) ~= 0 || any(diff(idx) ~= 1)
        error("Linear layer indices must be contiguous starting at 0 (found: %s).", mat2str(idx));
    end
    L = idx(end); % Last linear layer index

    layer_input = x_k;

    %% ----- Hidden Layers -----------------------------------------------
    for layer = 0:(L - 1)

        W = w.(['linear_' num2str(layer) '_weight']);
        b = w.(['linear_' num2str(layer) '_bias']);

        % Ensure bias is a column vector
        if isrow(b), b = b.'; end

        % Linear transform
        z = W * layer_input + b;

        % Nonlinearity (tanh)
        layer_input = tanh(z);

    end

    %% ----- Output Layer ------------------------------------------------
    W_out = w.(['linear_' num2str(L) '_weight']);
    b_out = w.(['linear_' num2str(L) '_bias']);

    % Ensure bias is a column vector
    if isrow(b_out), b_out = b_out.'; end

    y_k = W_out * layer_input + b_out;

end
