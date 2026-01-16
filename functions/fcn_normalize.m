function data_norm = fcn_Normalize(data, data_mean, data_std, inverse)
%FCN_NORMALIZE Z-score normalization (Simulink compatible)
%
% Applies feature-wise Z-score normalization or its inverse.
% Each feature is processed independently.
%
% data       : [N x F] input data (N samples, F features)
% data_mean  : [1 x F] mean value of each feature
% data_std   : [1 x F] standard deviation of each feature
% inverse    : 0 = normalize (z = (x - mean) / std)
%              1 = inverse normalization (x = z * std + mean)

    n_features = size(data,2);
    data_norm = zeros(size(data),'like',data);

    if inverse == 0
        for i = 1:n_features
            data_norm(:,i) = (data(:,i) - data_mean(1,i)) / data_std(1,i);
        end
    else
        for i = 1:n_features
            data_norm(:,i) = data(:,i) * data_std(1,i) + data_mean(1,i);
        end
    end
end
