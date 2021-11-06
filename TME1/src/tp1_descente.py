import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context

if __name__=='__main__':
    # Les données supervisées
    torch.manual_seed(0)
    x = torch.randn(50, 13)
    y = torch.randn(50, 3)

    # Les paramètres du modèle à optimiser
    w = torch.randn(13, 3)
    b = torch.randn(3)

    lin=Linear()
    mse=MSE()

    epsilon = 0.001
    writer = SummaryWriter()
    for n_iter in range(1000):

        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        ctxLin=Context()
        ctxMSE=Context()


        ##  TODO:  Calcul du forward (loss)
        loss=mse.forward(ctxMSE,lin.forward(ctxLin,x,w,b),y)

        writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        ##  TODO:  Calcul du backward (grad_w, grad_b)
        dyhat,dy=mse.backward(ctxMSE, torch.tensor(1.,dtype=torch.float64))
        _,dw,db=lin.backward(ctxLin,dyhat)

        ##  TODO:  Mise à jour des paramètres du modèle
        w=w-epsilon*dw
        b=b-epsilon*db


