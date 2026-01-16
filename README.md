# Neural Networks: Python Training → MATLAB Matrix Operations

Train neural networks in Python using [tsfast](https://github.com/daniel-om-weber/tsfast) (built on [fastai](https://github.com/fastai/fastai)), export them as JSON, and run forward passes in MATLAB using pure matrix operations, no toolboxes required.

## 🎯 Overview

This repository provides a complete workflow for:
1. **Training** neural networks in Python using tsfast/fastai
2. **Exporting** trained weights to JSON format
3. **Implementing** forward passes in MATLAB with transparent matrix operations

Supported architectures:
- **FNN** (Feedforward Neural Networks)
- **GRU** (Gated Recurrent Units)
- **LSTM** (Long Short-Term Memory)

## 💡 Use Cases

- **Model Predictive Control (MPC)** with neural network dynamics
- **Simulink deployment** without requiring latest toolbox versions
- **Embedded systems** needing compiled, toolbox-free code
- **Educational purposes** to understand network internals

This approach has been used, for example, in the implementation of a nonlinear MPC in C++ using CasADi, which was subsequently deployed as an S-function in MATLAB Real-Time.
The neural network was trained in Python and exported for efficient execution in a real-time environment:

> H. Schäfke, T.-L. Habich, C. Muhmann, S. F. G. Ehlers, T. Seel, and M. Schappler,  
> *Learning-Based Nonlinear Model Predictive Control of Articulated Soft Robots Using Recurrent Neural Networks*,  
> IEEE Robotics and Automation Letters, vol. 9, no. 12, pp. 11609–11616, Dec. 2024.  
> doi: 10.1109/LRA.2024.3495579  
> https://ieeexplore.ieee.org/abstract/document/10750121

The MPC implementation is publicly available [here](https://tlhabich.github.io/sponge/rnn_mpc/).

## 📁 Repository Structure

```
├── train_and_export_networks.ipynb  # Python: train & export models
├── models/                           # Exported JSON weights
│   ├── FNN_model_1layer.json
│   ├── FNN_model_nlayer.json
│   ├── GRU_model_1layer.json
│   ├── GRU_model_nlayer.json
│   ├── LSTM_model_1layer.json
│   └── LSTM_model_nlayer.json
├── functions/                        # MATLAB forward pass functions
│   ├── nn_fnn_1_layer.m
│   ├── nn_fnn_n_layer.m
│   ├── nn_gru_1_layer.m
│   ├── nn_gru_n_layer.m
│   ├── nn_lstm_1_layer.m
│   └── nn_lstm_n_layer.m
├── demo_fnn_forward.m               # MATLAB demo scripts
├── demo_gru_forward.m
└── demo_lstm_forward.m
```

## 🚀 Quick Start

### 1. Train Networks (Python)

```python
# Install dependencies
# Note: use "fastprogress==1.0.3"
pip install tsfast fastai torch

# Open and run train_and_export_networks.ipynb
# This trains FNN/GRU/LSTM models and exports weights to JSON
```


### 2. Run Forward Pass (MATLAB)

```matlab
% Load model and data
model = jsondecode(fileread("models/GRU_model_1layer.json"));
data = load("silverbox_data.mat");

% Initialize hidden state
h = zeros(model.meta.hidden_size, 1);

% Forward pass
for t = 1:length(data.X)
    x_t = data.X(t, :)';
    [y_t, h] = nn_gru_1_layer(x_t, h, model.weights);
end
```

Or simply run the demo scripts:
- `demo_fnn_forward.m`
- `demo_gru_forward.m`
- `demo_lstm_forward.m`

## 📊 Example Dataset

The repository uses the **Silverbox benchmark dataset** for system identification. It is a commonly used nonlinear dynamics dataset.