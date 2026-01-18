import torch
import numpy as np
import time
import import_ipynb
import torch.nn as nn
import pandas as pd 
from torch.distributions import Normal
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
import scipy.stats as scst
import pathlib,os
from scipy.special import expit
from sklearn import random_projection

from joblib import Parallel, delayed
import argparse
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

def sigmoid(z):
    return 1. / (1 + torch.exp(-z))

def log_gaussian(x, mu, sigma):
    """
        log pdf of one-dimensional gaussian
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma)
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)
def sv(d,dim):
    
   
        
    psi=0.3
    x0=multinomial.rvs(1,[psi**2, 2*psi*(1-psi), (1-psi)**2],size=(dim*d))
    Q=-1.0/np.sqrt(psi)*x0[:,0]+0*x0[:,1]+1.0/np.sqrt(psi)*x0[:,2]
    Q=Q.reshape((dim,d))        
    Q=gs(Q)
    Q=normalize_rows(Q)
    return Q.T

def project_x(Z):
    global A,V_train,V_test
    A=[]
    V_train=[]
    V_test=[]
    for j in range(M):
        MM=sv(p,Z.shape[1],METHOD_P)
        
        A.append(MM)
        V_train.append(np.dot(MM,Z.T).T)
        #V_test.append(np.dot(MM,Z_test.T).T)
def gs(x0):
    y0 = []
    for i in range(len(x0)):
        temp_vec = x0[i]
        for inY in y0 :
            proj_vec = proj(inY, x0[i])
            temp_vec = [item for item in map(lambda x,y: x-y, temp_vec, proj_vec)]
        y0.append(temp_vec)
    return normalize_rows(np.array(y0))
def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)
def gs_cofficient(v1, v2):
    return np.dot(v2, v1)/max(np.dot(v1, v1),1e-4)

def multiply(cofficient, v):
    return map((lambda x:x*cofficient), v)
def normalize_rows(x):
    arr=[]
    for row in x:
        lm=max(np.linalg.norm(row),1e-4)
        arr.append(row/lm)
    return np.array(arr)


class MLPLayer(nn.Module):
    
    """
        Layer of our BNN
    """
    def __init__(self, input_dim, output_dim, rho_prior, device, rho0=-6., lambda0=0.99):
        # initialize layers
        super(MLPLayer, self).__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initialize mu, rho and theta parameters for layer's weights
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(rho0, rho0))
        # initialize mu, rho and theta parameters for layer's biases, theta = logit(phi)
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(rho0, rho0))

        self.rho_prior = rho_prior
        self.device = device

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize log pdf of prior and vb distributions
        self.kl = 0

    def forward(self, X):
        
        """
            For one Monte Carlo sample
            :param X: [batch_size, input_dim]
            :return: output for one MC sample, size = [batch_size, output_dim]
        """
        # sample weights and biases
        sigma_w = torch.log(1 + torch.exp(self.w_rho))
        sigma_b = torch.log(1 + torch.exp(self.b_rho))
        sigma_prior = torch.log(1 + torch.exp(self.rho_prior))

        epsilon_w = Normal(0, 1).sample(self.w_mu.shape)
        epsilon_b = Normal(0, 1).sample(self.b_mu.shape)
        
        epsilon_w = epsilon_w.to(self.device)
        epsilon_b = epsilon_b.to(self.device)

        self.w = self.w_mu + sigma_w * epsilon_w
        self.b = self.b_mu + sigma_b * epsilon_b
        output = torch.mm(X, self.w) + self.b.expand(X.size()[0], self.output_dim)

        # record KL at sampled weight and bias
        kl_w = (torch.log(sigma_prior) - torch.log(sigma_w) +
                        0.5 * (sigma_w ** 2 + self.w_mu ** 2) / sigma_prior ** 2 - 0.5)

        kl_b = (torch.log(sigma_prior) - torch.log(sigma_b) +
                        0.5 * (sigma_b ** 2 + self.b_mu ** 2) / sigma_prior ** 2 - 0.5)

        self.kl = torch.sum(kl_w) + torch.sum(kl_b)

        return output
    
#### sparsefunc file content
class SFunc(nn.Module):
    #torch.set_default_dtype(torch.float32)
    """
        Our BNN
    """
    def __init__(self, data_dim, hidden_dim1, target_dim, device, sigma_noise=1):

        
        super(SFunc, self).__init__()
        self.rho_prior = torch.Tensor([np.log(np.exp(1.3) - 1)]).to(device)
        
        
        self.device = device

        self.l1 = MLPLayer(data_dim, hidden_dim1, self.rho_prior, self.device)
        

        self.l4 = MLPLayer(hidden_dim1, target_dim, self.rho_prior, self.device)

        self.target_dim = target_dim
        self.log_sigma_noise = torch.log(torch.Tensor([sigma_noise])).to(device)

    def forward(self, X):
        
        """
            output of the BNN for one Monte Carlo sample
            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """
        output = self.l1_sigmoid(self.l1(X))
#         output = self.l2_sigmoid(self.l2(output))
#         output = self.l3_sigmoid(self.l3(output))
        output = self.l4(output)
        return output

    def kl(self):
       
        kl = self.l1.kl +  self.l4.kl
        return kl

    def sample_elbo(self, X, y, n_samples, num_batches):
        
        """
            calculate the loss function - negative elbo
            :param X: [batch_size, data_dim]
            :param y: [batch_size, target_dim]
            :param n_samples: number of MC samples
            :param temp: temperature
            :return:
        """

        # initialization
        outputs = torch.zeros(n_samples, y.shape[0], self.target_dim).to(self.device)
        #print(outputs.shape)
        kls = 0.
        log_likes = 0.

        # make predictions and calculate prior, posterior, and likelihood for a given number of MC samples
         for i in range(n_samples):  # ith mc sample
            outputs[i] = self.forward(X)  # make predictions, (batch_size, target_dim)
            sample_kl = self.kl()  # get kl (a number)
            kls += sample_kl      
            # The following does not work when number of KL samples =1
            #log_likes += -F.cross_entropy(outputs[i].squeeze(), y, reduction='sum')
            log_likes += torch.sum((outputs[i].squeeze()-                        torch.log(torch.sum(torch.exp(outputs[i].squeeze()),dim=1)).view(-1,1)).index_select(-1,y).diag())
            # Use this when number of samples are larger than 1
             
            
        # calculate MC estimates of log prior, vb and likelihood
        kl_MC = kls/float(n_samples)
        # calculate negative loglikelihood
        nll_MC = - log_likes/float(n_samples)

        # calculate negative elbo
        loss = kl_MC / num_batches + nll_MC
        return loss, outputs.squeeze()
    
def project_x_python(l,u):
    global V_train,V_test,Y_train,Y_test,P_list
    
    train_dataset = datasets.MNIST(root='./mnist_data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./mnist_data/',
                                  train=False,
                                  transform=transforms.ToTensor())
    
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=60000,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000,
                                          shuffle=False)
    
    for batch in train_loader:
        X_train=np.array(batch[0]).reshape(60000,28*28)
        Y_train=np.array(batch[1])
        y_train = torch.Tensor(Y_train).type(torch.LongTensor)
    
        
    for batch in test_loader:
        
        X_test=np.array(batch[0]).reshape(10000,28*28)
        Y_test=np.array(batch[1])
        y_test = torch.Tensor(Y_test).type(torch.LongTensor)
    

    V_train=[]
    Y_train=[]
    V_test=[]
    Y_test=[]
    P_list=[]
    
    
    X=np.concatenate((X_train,X_test),axis=0)
    for j in range(M):
        p=np.random.randint(l,u)
        MM=sv(p,X.shape[1])  
        X_new=np.dot(MM,X.T).T
        x_train=X_new[0:60000,:]
        x_test=X_new[60000:,:]



        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        V_train.append(x_train)
        V_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)
        P_list.append(p)
            
            
def parallel(j):
    torch.set_default_dtype(torch.float32)
    
    global x_train,x_test,y_train,y_test,data_dim,hidden_dim1
    
    x_train,x_test,y_train,y_test=V_train[j],V_test[j],Y_train[j],Y_test[j]
    data_dim=x_train.shape[1]
    
    loss1,train_accur,test_accur,loss,pred_train,pred_test=train_model(x_train,x_test,y_train,y_test) 
    
    model=[loss1,train_accur,test_accur,loss,pred_train,pred_test]  
    
    
    return model

def train_model(x_train,x_test,y_train,y_test):
    #global labels,labels1
    
    torch.set_default_dtype(torch.float64)
    
    net = SFunc(data_dim=data_dim, hidden_dim1 = hidden_dim1, target_dim = target_dim, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    
    #torch.set_default_dtype(torch.float64)
    LS=[]
    PD_TR=[]
    PD_TT=[]
    
    
    
    for epoch in range(epochs): 
        
        permutation = torch.randperm(V_train[0].shape[0])
        for i in range(0, V_train[0].shape[0], batch_size):
            
            indices = permutation[i:i + batch_size]
            
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)


            loss, preds = net.sample_elbo(batch_x.double(), batch_y, 1, num_batches)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            
        
        
 
            
        
        
        if epoch%jump==0 or epoch>(epochs-100):
            loss1, pred = net.sample_elbo(x_train.double(), y_train, 30, 1)
            #_, pred = net.sample_elbo(x_train, y_train, 5, num_batches)
            #print(pred.shape)
            pred = pred.mean(dim=0)
            pred1 = torch.max(pred, 1)[1].to(device)
            
            train_accur = np.mean(pred1.detach().numpy() == y_train.detach().numpy())
            
            
            
            _, pred2 = net.sample_elbo(x_test, y_test, 30, 1)
            pred21 = pred2.mean(dim=0)
            pred2 = torch.max(pred21, 1)[1].to(device)
            test_accur =  np.mean(pred2.detach().numpy() == y_test.detach().numpy())
            
            if epoch>(epochs-100):
                LS.append(loss1.detach().numpy())
                PD_TR.append(pred.detach().numpy())
                PD_TT.append(pred21.detach().numpy()) 
            
            if epoch%jump==0:
                print('epoch: {}, loss: {}, train acc: {} test acc: {}'.format(epoch,loss1,train_accur,test_accur))

    LS=np.array(LS)    
    PD_TR=np.array(PD_TR)
    PD_TT=np.array(PD_TT)
    ls=np.mean(LS)
    
    pd_tr=np.mean(PD_TR,0)
    pd_tt=np.mean(PD_TT,0)
    return loss1,train_accur,test_accur,ls,pd_tr,pd_tt


    

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default = 500, help='What is the the number of epochs')
parser.add_argument('--l', type=int, default = 580, help='what is the lower projected dim')
parser.add_argument('--u', type=int, default = 600, help='what is the upper projected dim')
parser.add_argument('--M', type=int, default = 128, help='what is the M')
parser.add_argument('--nodes', type=int, default = 64, help='what is the number of nodes in hidden layer')
parser.add_argument('--lr', type=float, default = 0.01 , help='what is the learning rate')
parser.add_argument('--batch', type=int, default = 512 , help='what is the batch size')
parser.add_argument('--jump', type=int, default = 10 , help='what is the jump size')

args = parser.parse_args()

if __name__ == '__main__':
    
    device = torch.device('cpu')

    np.random.seed(123)
    torch.manual_seed(456)
    M=args.M
    
    

    project_x_python(args.l,args.u)
    batch_size = args.batch

    num_batches = V_train[0].shape[0] / batch_size
    
    
    learning_rate = args.lr
    epochs = args.epochs
    hidden_dim1 = args.nodes
    jump = args.jump
 
    target_dim = 10
    L = 1
   
    
    output = Parallel(n_jobs=2)(delayed(parallel)(i) for i in range(M))
    
    output1=output
    
    output=[output1[i][3] for i in range(M)]
    
    
    P=torch.Tensor((1.0/M)*np.ones((M,)))
    Q1=-1.0*torch.Tensor(np.array(output))+torch.log(P)
    Q=nn.Softmax(dim=0)(Q1)
    
    loss_overall=(torch.sum(Q[Q>0]*(-Q1[Q>0]+torch.log(Q[Q>0])))).detach().numpy()
    
    
    
    pred_tr=np.mean(np.array([output1[i][4]*Q[i].detach().numpy() for i in range(M)]),0)
    
    
    pred_tr=np.max(pred_tr, 1)[1].astype(int)
    
    train_accur =  np.mean(pred_tr == Y_train[0].detach().numpy())    
    
    
    pred_tt=np.mean(np.array([output1[i][5]*Q[i].detach().numpy() for i in range(M)]),0)
    pred_tt=np.max(pred_tt, 1)[1].astype(int)
    test_accur =  np.mean(pred_tt == Y_test[0].detach().numpy())
    
    
    result=np.concatenate((np.array(P_list).reshape(M,1),np.array(Q).reshape(M,1),np.array(output).reshape(M,1)),axis=1)
   
    
    

    for i in range(len(output)):
        print('Q: {}, P: {}, , loss: {}, train acc: {}, test acc: {}'.format(Q[i],P_list[i],output1[i][0],output1[i][1],output1[i][2]))
        
    print('Overall Results: loss: {}, train acc: {}, test acc: {}'.format(loss_overall,train_accur,test_accur))  
    
   
        