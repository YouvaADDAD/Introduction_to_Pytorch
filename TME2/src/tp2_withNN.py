import datamaestro
from torch.nn import Linear,Tanh,MSELoss,Sequential
import torch
from Module import normalise_data_min_max,normalise_data_mean_std,generator
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


def Multiple_couche(datax,datay,testx,testy,writer,batch_size=600,NB_EPOCH=10000, eps=1e-3,hidden=5):
    datax, datay = normalise_data_mean_std(datax, datay)
    testx,testy=normalise_data_mean_std(testx, testy)
    lin1=Linear(datax.shape[1],hidden)
    tan=Tanh()
    lin2=Linear(hidden,datay.shape[1])
    mse=MSELoss()
    optim = torch.optim.SGD(params=[lin1.weight,lin1.bias,lin2.weight,lin2.bias], lr=eps)
    optim.zero_grad()
    for i in range(NB_EPOCH):
        for batch_x, batch_y in generator(datax, datay, batch_size):
            reslin1 = lin1(batch_x)
            restan=tan(reslin1)
            reslin2=lin2(restan)
            loss=mse(reslin2,batch_y)
            loss.backward()
            print(loss)   
            writer.add_scalar('Loss/train', loss.item(), i)
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            reslin1 = lin1(testx)
            restan=tan(reslin1)
            reslin2=lin2(restan)
            loss=mse(reslin2,testy)
            writer.add_scalar('Loss/test', loss.item(), i)


def Multiple_Couch_container(datax,datay,testx,testy,writer,batch_size=600,NB_EPOCH=10000, eps=1e-3,hidden=5):
    datax, datay = normalise_data_mean_std(datax, datay)
    testx,testy  =normalise_data_mean_std(testx, testy)
    model=Sequential(Linear(datax.shape[1],hidden),Tanh(),Linear(hidden,datay.shape[1]))
    mse=MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=eps)
    optim.zero_grad()
    for i in range(NB_EPOCH):
        for batch_x, batch_y in generator(datax, datay, batch_size):
            loss=mse(model(batch_x),batch_y)
            loss.backward()
            print(loss)
            writer.add_scalar('Loss/train', loss.item(), i)
            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            loss=mse(model(testx),testy)
            writer.add_scalar('Loss/test', loss.item(), i)


if __name__=='__main__':
    torch.manual_seed(0)
    data = datamaestro.prepare_dataset("edu.uci.boston")
    colnames, datax, datay = data.data()
    datax = torch.tensor(datax, dtype=torch.float)
    datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)
    writer = SummaryWriter()
    X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.33, random_state=0)
    Multiple_Couch_container(X_train, y_train,X_test,y_test,writer,batch_size=1,NB_EPOCH=2000)

