%% FNN Forward Pass Demo
% Simple demonstration of FNN network forward propagation

clear; clc;
addpath("functions");

dataFile   = "silverbox_data.mat";
modelFile1 = "models/FNN_model_1layer.json";
modelFile3 = "models/FNN_model_3layer.json";

%% Load data (X: [T x input_dim], Y: [T x output_dim])
data = load(dataFile);
X = data.X; % inputs
Y = data.Y; % ground truth

numTimesteps = size(X, 1);

%% ===================== FNN with 1 layer =====================
%% Load model (JSON)
model1 = jsondecode(fileread(modelFile1));

mean = model1.meta.norm.mean;
std  = model1.meta.norm.std;

%% Preallocate outputs and initialize hidden state
Y_pred_1 = zeros(numTimesteps, model1.meta.output_size);

%% Forward pass through time
for k = 1:numTimesteps
    % Current input as column vector [input_dim, 1]
    xk = X(k, :).';

    % Normalize input
    xk = fcn_normalize(xk, mean, std, 0);

    % 1-layer FNN forward step
    yk = nn_fnn_1_layer(xk, model1.weights);

    % Store prediction
    Y_pred_1(k, :) = yk.';
end

%% ===================== FNN with 3 layers =====================
%% Load model (JSON)
model3 = jsondecode(fileread(modelFile3));

%% Preallocate outputs and initialize hidden state
Y_pred_3 = zeros(numTimesteps, model3.meta.output_size);

%% Forward pass through time
for k = 1:numTimesteps
    % Current input as column vector [input_dim, 1]
    xk = X(k, :).';

    % Normalize input
    xk = fcn_normalize(xk, mean, std, 0);

    % 3-layer FNN forward step
    yk = nn_fnn_n_layer(xk, model3.weights);

    % Store prediction
    Y_pred_3(k, :) = yk.';
end

%% Plot results
plotStart = 1;
plotEnd   = min(500, numTimesteps);

t = plotStart:plotEnd;

figure("Name","FNN Forward Pass Demo");

% --- Top subplot: outputs ---
subplot(2,1,1);
plot(t, Y(t,:), "k", "LineWidth", 1.5); hold on;
plot(t, Y_pred_1(t,:), "--", "LineWidth", 1.5);
plot(t, Y_pred_3(t,:), ":", "LineWidth", 1.5);
legend("Ground Truth", "FNN 1 Layer", "FNN 3 Layer", "Location", "best");
ylabel("Output");
title("Predictions vs Ground Truth");
grid on;

% --- Bottom subplot: inputs ---
subplot(2,1,2);
plot(t, X(t,:), "LineWidth", 1.0);
xlabel("Time Step");
ylabel("Input");
title("Input Signals");
grid on;
