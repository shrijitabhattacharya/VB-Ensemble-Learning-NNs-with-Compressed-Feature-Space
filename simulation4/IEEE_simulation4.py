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
import argparse
from joblib import Parallel, delayed
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from scipy.linalg import svd
from scipy.stats import multinomial
from sklearn.metrics import roc_curve, auc
from scipy.special import gammainc
from sklearn.utils.validation import check_random_state

def sigmoid(z):
    return 1. / (1 + torch.exp(-z))

def log_gaussian(x, mu, sigma):
    """
        log pdf of one-dimensional gaussian
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma)
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)

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

def normalize(v):
    return v / max(np.sqrt(v.dot(v)),1e-4)

def orthonormalize(A):
    n = A.shape[1]
    A[:, 0] = normalize(A[:, 0])
    for i in range(1, n):
        Ai = A[:, i]
        for j in range(0, i):
            Aj = A[:, j]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        A[:, i] = normalize(Ai)
    return A



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
        V_test.append(np.dot(MM,Z_test.T).T)
        
def data_hyperball(d1=500,m=10):
    X = np.zeros((300, d1))
    #X=np.random.rand(300,d1)*0.001
    X[:, :m] = hyperBall(n=300, d=m, radius=1, random_state=0)
    return X

def hyperBall(n, d, radius=1.0, center=[], random_state=None):
    """
    Generates a sample from a uniform distribution on the hyperball

    Parameters
    ----------
    n: int 
        Number of data points.
    d: int
        Dimension of the hyperball
    radius: float
        Radius of the hyperball
    center: list, tuple, np.array
        Center of the hyperball
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    random_state_ = check_random_state(random_state)
    if center == []:
        center = np.array([0] * d)
    r = radius
    x = random_state_.normal(size=(n, d))
    ssq = np.sum(x ** 2, axis=1)
    fr = r * gammainc(d / 2, ssq / 2) ** (1 / d) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, d))
    p = center + np.multiply(x, frtiled)
    return p

class MLPLayer(nn.Module):
    """
        Layer of our BNN
    """
    def __init__(self, input_dim, output_dim, rho_prior, device, lambda0=0.99):
        # initialize layers
        super(MLPLayer, self).__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        
       
        
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.6, 0.6))
        self.w_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(rho_value, rho_value))

        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.6, 0.6))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(rho_value, rho_value))

        self.rho_prior = rho_prior
        self.device = device

        
        self.w = None
        self.b = None

        
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
#### sparsefunc file content
class SFunc(nn.Module):
    """
        Our BNN
    """
    def __init__(self, data_dim, hidden_dim1, target_dim, device, sigma_noise=1):

        # initialize the network using the MLP layer
        super(SFunc, self).__init__()
        self.rho_prior = torch.Tensor([np.log(np.exp(1.3)-1)]).to(device)
        self.device = device

        self.l1 = MLPLayer(data_dim, hidden_dim1, self.rho_prior, self.device)
        self.l1_sigmoid = nn.Sigmoid()
#         self.l2 = MLPLayer(hidden_dim1, hidden_dim2, self.rho_prior, self.device)
#         self.l2_sigmoid = nn.Sigmoid()
#         self.l3 = MLPLayer(hidden_dim2, hidden_dim3, self.rho_prior, self.device)
#         self.l3_sigmoid = nn.ReLU()
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
        #output = nn.Sigmoid()(self.l4(output))
        return output

    def kl(self):
        # calculate the kl over all the layers of the BNN
        kl = self.l1.kl +  self.l4.kl
        return kl

    def sample_elbo(self, X, y, n_samples, num_batches):
        """
            calculate the loss function - negative elbo
            :param X: [batch_size, data_dim]
            :param y: [batch_size]
            :param n_samples: number of MC samples
            :return:
        """

        # initialization
        outputs = torch.zeros(n_samples, y.shape[0], self.target_dim).to(self.device)
        #print(outputs.shape)
        kls = 0.
        log_likes = 0.

        # make predictions and calculate prior, posterior, and likelihood for a given number of MC samples
        for i in range(n_samples):  # ith mc sample
            outputs[i] = self.forward(X)
            #y=y.double()# make predictions, (batch_size, target_dim)
            sample_kl = self.kl()  # get kl (a number)
            kls += sample_kl      
            # The following does not work when number of KL samples =1
            #log_likes += -F.cross_entropy(outputs[i].squeeze(), y, reduction='sum')
            log_likes += torch.sum(y*outputs[i].squeeze()-torch.log(1+torch.exp(outputs[i].squeeze())))
            #log_likes=log_likes+nn.BCELoss()(outputs[i].squeeze(),y)
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
    
    
    
        
    new_data=data_hyperball()

    beta=np.random.uniform(-0.5,0.5,500)
    y=np.sin(np.dot(new_data,beta))-np.exp(np.dot(new_data,beta))+1.05*np.cos(np.dot(new_data,beta))
    Y_til=np.where(y>0,1,0)
    X_orig=new_data
        
    

       
    X_til=scale(X_orig,with_mean=True,with_std=True)  
    V_train=[]
    Y_train=[]
    V_test=[]
    Y_test=[]
    P_list=[]
 
    for j in range(M):
        p=np.random.randint(l,u)
        MM=sv(p,X_til.shape[1])
        X_til_new=np.dot(MM,X_til.T).T

        x_train, x_test, y_train, y_test = train_test_split(X_til, Y_til, test_size=0.3, random_state=split_state) 

        x_train = torch.Tensor(x_train)
        y_train = torch.flatten(torch.Tensor(y_train)).type(torch.LongTensor)
        x_test = torch.Tensor(x_test)
        y_test = torch.flatten(torch.Tensor(y_test)).type(torch.LongTensor)
        V_train.append(x_train)
        V_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)
        P_list.append(p)

    
    
        

            
def parallel(j):
   
    
    global x_train,x_test,y_train,y_test,data_dim,hidden_dim1
    
    
    x_train,x_test,y_train,y_test=V_train[j],V_test[j],Y_train[j],Y_test[j]
    
    data_dim=x_train.shape[1]
    loss1,train_accur,test_accur,loss,pred_train,pred_test=train_model(x_train,x_test,y_train,y_test) 
    
    model=[loss1,train_accur,test_accur,loss,pred_train,pred_test]  
    
    return model

def train_model(x_train,x_test,y_train,y_test):
   
    
    torch.set_default_dtype(torch.float64)
    
    net = SFunc(data_dim=data_dim, hidden_dim1 = hidden_dim1, target_dim = target_dim, device=device).to(device)
    
   
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
   
    
    LS=[]
    PD_TR=[]
    PD_TT=[]
    for epoch in range(epochs): 
        
        permutation = torch.randperm(x_train.shape[0])
        
        for i in range(0, x_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)        
            loss, preds1 = net.sample_elbo(batch_x, batch_y, sample_size, num_batches)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            

        if epoch%jump==0 or epoch>(epochs-100):
            loss1, pred = net.sample_elbo(x_train, y_train,30,1)
            pred=pred.mean(dim=0)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            _, pred2 = net.sample_elbo(x_test, y_test,30,1)
            pred2 =pred2.mean(dim=0)
            if epoch>(epochs-100):
                LS.append(loss1.detach().numpy())
                PD_TR.append(pred.detach().numpy())
                PD_TT.append(pred2.detach().numpy()) 
            pred = (pred>0).long()
            train_accur = np.mean(pred.detach().numpy() == y_train.detach().numpy())
            pred2 = (pred2>0).long()
            
            test_accur =  np.mean(pred2.detach().numpy() == y_test.detach().numpy())
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
parser.add_argument('--epochs', type=int, default = 2000, help='What is the the number of epochs')
parser.add_argument('--l', type=int, default = 8, help='what is the lower projected dim')
parser.add_argument('--u', type=int, default =17, help='what is the upper projected dim')
parser.add_argument('--M', type=int, default =32 , help='what is the M')
parser.add_argument('--sample', type=int, default = 1, help='what is the sample size')
parser.add_argument('--nodes', type=int, default =64, help='what is the number of nodes in hidden layer')
parser.add_argument('--lr', type=float, default = 0.01 , help='what is the learning rate')
parser.add_argument('--jump', type=int, default = 1000, help='what is the jump size')



args = parser.parse_args()

if __name__ == '__main__':
    
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float64)
   
    M=args.M
    
    
    sample_size=args.sample
    
    
    rho_value=-6.0
    for k in range(12,22,10):
        split_state=k
        project_x_python(args.l,args.u)

        batch_size=V_train[0].shape[0]

        num_batches = V_train[0].shape[0] / batch_size


        learning_rate = args.lr
        epochs = args.epochs
        hidden_dim1 = args.nodes
        jump = args.jump

        target_dim = 1
        L = 1

        output = Parallel(n_jobs=2)(delayed(parallel)(i) for i in range(M))

        output1=output

        output=[output1[i][3] for i in range(M)]


        P=torch.Tensor((1.0/M)*np.ones((M,)))
        Q1=-1.0*torch.Tensor(np.array(output))+torch.log(P)
        Q=nn.Softmax(dim=0)(Q1)

        loss_overall=(torch.sum(Q[Q>0]*(-Q1[Q>0]+torch.log(Q[Q>0])))).detach().numpy()

        pred_tr=np.mean(np.array([output1[i][4]*Q[i].detach().numpy() for i in range(M)]),0)
        pred_tr=(pred_tr>0).astype(int)
        train_accur =  np.mean(pred_tr == Y_train[0].detach().numpy())    


        pred_tt=np.mean(np.array([output1[i][5]*Q[i].detach().numpy() for i in range(M)]),0)

        pred_pp=1.0/(1.0+np.exp(-pred_tt))

        pred_tt=(pred_tt>0).astype(int)


        test_accur =  np.mean(pred_tt == Y_test[0].detach().numpy())

        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test[0].detach().numpy(), pred_pp)

        roc_auc= auc(false_positive_rate, true_positive_rate)


        result=np.concatenate((np.array(P_list).reshape(M,1),np.array(Q).reshape(M,1),np.array(output).reshape(M,1)),axis=1)
        for i in range(len(output)):
            print('Q: {}, P: {}, , loss: {}, train acc: {}, test acc: {}'.format(Q[i],P_list[i],output1[i][0],output1[i][1],output1[i][2]))

        print('Overall Results: loss: {}, train acc: {}, test acc: {}, AUC: {}'.format(loss_overall,train_accur,test_accur,roc_auc))  
        result1=np.concatenate((loss_overall.reshape(1,-1),train_accur.reshape(1,-1),test_accur.reshape(1,-1)),axis=1)

    


