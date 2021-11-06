import torch

from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Module import *


# TODO: 
if __name__=='__main__':
    torch.manual_seed(0)
    writer = SummaryWriter()
    data = datamaestro.prepare_dataset("edu.uci.boston")
    colnames, datax, datay = data.data()
    datax = torch.tensor(datax, dtype=torch.float)
    datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.33, random_state=0)
    train_test(X_train,y_train,X_test,y_test,writer,batch_size=1)