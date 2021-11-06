# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch as tr
from torch.autograd import Function
from torch.autograd import gradcheck
from torch.nn import Linear


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        assert(yhat.shape==y.shape)
        #  TODO:  Renvoyer la valeur de la fonction
        return tr.sum((yhat-y)**2)/y.nelement()


    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        assert (yhat.shape == y.shape)
        dyhat= 2*(yhat-y)*grad_output/yhat.nelement()
        dy= -2*(yhat-y)*grad_output/yhat.nelement()
        return dyhat,dy



#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    """ Linear qui applique X.W+B"""

    @staticmethod
    def forward(ctx, X,W,b):
        ctx.save_for_backward(X,W,b)
        assert(len(X.shape)==2)
        assert(X.shape[1]==W.shape[0])
        return X@W+b

    @staticmethod
    def backward(ctx, grad_output):
        X,W,b=ctx.saved_tensors
        dx=grad_output@ W.t()
        dw=X.t() @ grad_output
        db= grad_output.sum(keepdim=True,axis=0)
        return dx,dw,db









## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

