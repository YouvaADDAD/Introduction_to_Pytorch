import torch
import torch as tr


def mse (yhat,y):
    return tr.mean((yhat-y)**2)

def linear(X,W,b):
    return torch.addmm(b,X,W)


def normalise_data_min_max(datax,datay):
    min_val = datax.min()
    max_val = datax.max()
    datax = (datax - min_val) / (max_val - min_val)
    return datax,datay

def normalise_data_mean_std(datax,datay):
    mean=datax.mean(dim=0)
    std=datax.std(dim=0)
    datax=(datax-mean)/std
    return datax,datay


def generator(datax,datay,batch_size):
    length = len(datax)
    indices = tr.randperm(length)
    n_split = length // batch_size
    if (length % batch_size != 0):
        n_split += 1
    for i in range(n_split):
        indexes = indices[i * batch_size:(i + 1) * batch_size]
        yield datax[indexes], datay[indexes]


def train_test(datax,datay,testx,testy,writer,batch_size=600,maxIter=10000, eps=1e-3):
    W=tr.randn([datax.shape[1],datay.shape[1]],requires_grad=True)
    b=tr.randn(datay.shape[1],requires_grad=True)
    datax,datay=normalise_data_mean_std(datax,datay)
    testx,testy=normalise_data_mean_std(testx,testy)
    for i in range(maxIter):
        for batch_x, batch_y in generator(datax,datay,batch_size):
            loss=mse(linear(batch_x,W,b), batch_y)
            print(loss)
            loss.backward()
            #Faire attention a mettre a jour w et b en place
            #et utilisé le no_grad pour ne pas fausser le track d'autograd
            with torch.no_grad():
                # Inplace changes
                W.sub_(W.grad * eps)
                b.sub_(b.grad * eps)
                W.grad.zero_()
                b.grad.zero_()
            writer.add_scalar('Loss/train', loss, i)
        with torch.no_grad():
            loss=mse(linear(testx,W,b), testy)
            writer.add_scalar('Loss/test', loss, i)


        #with torch.no_grad():
        #    loss = mse(linear(testx, W, b), testy)
        #    writer.add_scalar('Loss/test', loss, i)










