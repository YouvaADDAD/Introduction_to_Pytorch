import datamaestro
from torch.nn import Linear,Tanh,MSELoss
import torch
from Module import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


if __name__=='__main__':
    torch.manual_seed(0)
    EPS = 1e-3
    NB_EPOCH = 1000
    data = datamaestro.prepare_dataset("edu.uci.boston")
    colnames, datax, datay = data.data()
    datax = torch.tensor(datax, dtype=torch.float)
    datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)
    datax, testx, datay, testy = train_test_split(datax, datay, test_size=0.33, random_state=0)
    batch_size=1
    datax,datay=normalise_data_mean_std(datax,datay)
    testx,testy=normalise_data_mean_std(testx,testy)
    writer = SummaryWriter()
    W = torch.randn([datax.shape[1], datay.shape[1]], requires_grad=True)
    b = torch.randn(datay.shape[1], requires_grad=True)
    optim = torch.optim.SGD(params=[W, b], lr=EPS)  ## on optimise selon w et b, lr : pas de gr
    optim.zero_grad()
    # Reinitialisation du gradient
    for i in range(NB_EPOCH):
        for batch_x, batch_y in generator(datax, datay, batch_size):
            loss = mse(linear(batch_x, W, b), batch_y)  # Calcul du cout
            loss.backward()
            writer.add_scalar('Loss/train', loss, i)
            optim.step()
            optim.zero_grad()
        with torch.no_grad():
            loss=mse(linear(testx,W,b), testy)
            print(loss)
            writer.add_scalar('Loss/test', loss, i)
