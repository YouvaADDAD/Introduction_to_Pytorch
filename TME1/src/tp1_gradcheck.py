import torch
from tp1 import mse, linear

# Test du gradient de MSE


if __name__=='__main__':
    yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
    y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
    print(torch.autograd.gradcheck(mse, (yhat, y)))

    #  TODO:  Test du gradient de Linear (sur le même modèle que MSE)

    X=torch.randn(100,34, requires_grad=True, dtype=torch.float64)
    W=torch.randn(34,12, requires_grad=True, dtype=torch.float64)
    b=torch.randn(1,12, requires_grad=True, dtype=torch.float64)
    print(torch.autograd.gradcheck(linear, (X,W,b)))

