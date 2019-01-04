"""
Fitting a random data using fully connected network with one hidden layer.
"""


import torch


dtype = torch.float
device = torch.device('gpu:0' if torch.cuda.is_available() else 'cpu')

# Define dimensions
N, D_IN, H, D_OUT = 64, 1000, 100, 10

# Create random data
X = torch.randn(N, D_IN, device=device, dtype=dtype)
y = torch.randn(N, D_OUT, device=device, dtype=dtype)

# Initialize weights randomly
w_1 = torch.randn(D_IN, H, device=device, dtype=dtype)
w_2 = torch.randn(H, D_OUT, device=device, dtype=dtype)

# Define learning rate
lr = 1e-6

# Loop over number of epochs to fit the data to nn
for epoch in range(50):
    # Forward pass
    h = X.mm(w_1)
    h_relu = torch.clamp(h, 0)
    y_pred = h_relu.mm(w_2)

    # Compute loss
    loss = (y_pred - y).pow(2).sum().item()
    print(f'Loss {epoch} : {loss:.4f}')

    # Backward pass
    y_pred_grad = 2 * (y_pred - y)
    w_2_grad = h_relu.t().mm(y_pred_grad)
    h_relu_grad = y_pred_grad.mm(w_2.t())
    h_grad = h_relu_grad.clone()
    h_grad[h < 0] = 0
    w_1_grad = X.t().mm(h_grad)

    # Update wights using gradient descent
    w_1 -= lr * w_1_grad
    w_2 -= lr * w_2_grad
