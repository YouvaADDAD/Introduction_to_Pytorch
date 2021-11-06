from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
# Téléchargement des données
from datamaestro import prepare_dataset

#  TODO: 
class State :
    def __init__(self ,model,optim):
        self .model = model
        self .optim = optim
        self .epoch , self . iteration = 0,0

class DataCreator(Dataset):
    def __init__(self,X,y):
        self.length=len(X)
        self.datax=torch.tensor(X).view(len(X),-1)/255 #Pour la normalisation et vectorization
        self.datay=torch.tensor(y)

    def __getitem__( self ,index ):
        return self.datax[index], self.datay[index]

    def __len__(self):
        return self.length

class AutoEncoder(torch.nn.Module):
    def __init__(self,in_size,hidden_size=100):
        super(AutoEncoder,self).__init__()
        lin1=nn.Linear(in_size,hidden_size)
        lin2=nn.Linear(hidden_size,in_size)
        lin2.weight = nn.Parameter(lin1.weight.t())
        self.Encoder=nn.Sequential(lin1,nn.ReLU())
        self.Decoder=nn.Sequential(lin2,nn.Sigmoid())
        self.autoEncoder=nn.Sequential(self.Encoder,self.Decoder)

    def forward(self,input):
        return self.autoEncoder(input)


class AutoEncoder2(torch.nn.Module):
    def __init__(self,in_size):
        super(AutoEncoder2,self).__init__()
        lin1=nn.Linear(in_size,100)
        lin2=nn.Linear(100,10)
        lin3=nn.Linear(10,100)
        lin4=nn.Linear(100,in_size)

        #Share Weight
        lin3.weight=nn.Parameter(lin2.weight.t())
        lin4.weight = nn.Parameter(lin1.weight.t())

        #Model
        self.Encoder=nn.Sequential(lin1,nn.Tanh(),lin2,nn.Tanh())
        self.Decoder=nn.Sequential(lin3,nn.Tanh(),lin4,nn.Sigmoid())
        self.autoEncoder=nn.Sequential(self.Encoder,self.Decoder)

    def forward(self,input):
        return self.autoEncoder(input)

class HighwayNetwork(torch.nn.Module):
    def __init__(self,in_size,n_layer=3):
        super(HighwayNetwork,self).__init__()
        self.n_layer=n_layer
        self.gate  = nn.ModuleList([nn.Linear(in_size,in_size) for _ in range(n_layer)])
        self.layers= nn.ModuleList([nn.Linear(in_size,in_size) for _ in range(n_layer)])

    def forward(self,input):
        for i in range(self.n_layer):
            layer_res=nn.Sigmoid()
            layer_res=layer_res(self.layers[i](input))
            gate=nn.Sigmoid()
            gate=gate(self.gate[i](input))
            output=gate*layer_res+(1-gate)*input
        return output


if __name__=='__main__':
    ds = prepare_dataset("com.lecun.mnist");
    train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()
    # Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Pour visualiser
    # Les images doivent etre en format Channel (3) x Hauteur x Largeur
    #images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
    # Permet de fabriquer une grille d'images
    #images = make_grid(images)
    # Affichage avec tensorboard
    #writer.add_image(f'samples', images, 0)

    savepath = Path("model.pch")


    #Learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPS=10e-3
    BATCH_SIZE = 256
    NB_EPOCH=1000
    in_size=train_images.shape[1]*train_images.shape[2]

    #DataSet train
    data = DataLoader(DataCreator(train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)

    #Chargement si sauvgarde il y a
    if savepath. is_file ():
        with savepath.open("rb") as fp:
            state = torch.load(fp)
    #Model d'apprentissage avec BCELoss
    else :
        model=HighwayNetwork(in_size,n_layer=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=EPS)
        state = State(model,optimizer)
    
    loss_fn = nn.MSELoss()

    torch.manual_seed(0)
    state.model.to(device)

    #Training

    for i in range(NB_EPOCH):
        for image_batch, _ in data:
            yhat=state.model(image_batch)
            loss = loss_fn(yhat, image_batch)
            state.optim.zero_grad()
            loss.backward()
            state.optim.step()
            print(f'loss at epoch {i}->{loss}')
            state.iteration += 1
            writer.add_scalar('Loss/train', loss.item(), i)

            #Sauvegarde du modéle
            with savepath.open("wb") as fp:
                state.epoch = i+1 
                torch.save(state ,fp)
        if i%10==0:
            with torch.no_grad():
                images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()
                images = make_grid(images)
                writer.add_image(f'samples', images, i)

                test=torch.tensor(train_images[0:8]).float().view(len(train_images[0:8]),-1)
                imagesRecon=state.model(test)
                imagesRecon = imagesRecon.view(-1,train_images.shape[1],train_images.shape[2]).unsqueeze(1).repeat(1,3,1,1).double()
                imagesRecon=make_grid(imagesRecon)
                writer.add_image(f'reconstructed', imagesRecon, i)



    
