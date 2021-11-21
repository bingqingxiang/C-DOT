import numpy as np
import math
import pandas as pd  
import os
import pickle

def create_centralized_data(dataDimension,TotalSamples,top_rank,eigen_gap):
    """ Generate synthetic data """
    diag = np.zeros((dataDimension, dataDimension))
    a=np.linspace(1,0.8,top_rank)
    b=np.linspace(0.8*eigen_gap,0.1,dataDimension-top_rank)
    np.fill_diagonal(diag,np.concatenate((a,b)))  
    
    data0 = np.random.randn(dataDimension,TotalSamples)
    data=np.matmul(diag,data0)
    return data,TotalSamples,dataDimension

def distributed_data(datasets,data,number_of_nodes,TotalSamples):
    """ Column-wise distribute data, split data into (number_of_nodes) files"""
    if not os.path.exists('Data/Test/{}'.format(datasets)):
        os.mkdir('Data/Test/{}'.format(datasets))
        
    s = math.floor(TotalSamples /number_of_nodes)
    for i in range(number_of_nodes):       # loop for nodes
        Yi = data[:,i*s:(i+1)*s]
        with open("Data/Test/{}/data{}.pickle".format(datasets,i), 'wb') as handle:
                pickle.dump(Yi, handle)
                
def load_local_data(filename,node_id):#name of the file, id of the current node
    """ Take name of the file folder and and node id to load the dataset for 1 node, and return it's number of samples,and it's data dimension"""
    data=pd.read_pickle("Data/Test/{}/data{}.pickle".format(filename,node_id))
    
    dataDimension=data.shape[0]
    NumSamples=data.shape[1]

    return data,NumSamples,dataDimension

def distributed_covariance(data,number_of_nodes,TotalSamples):
     """ Column-wise distribute data, split data and put sample covariance into (number_of_nodes) files"""
    C = np.zeros((number_of_nodes,), dtype=np.object)
    s = math.floor(TotalSamples /number_of_nodes)
    for i in range(number_of_nodes):       # loop for nodes
        Yi = data[:,i*s:(i+1)*s]
        C[i] = (1/(s))*np.dot(Yi,Yi.transpose())
        #np.save("Data/Test/data_covariance_{}".format(i),C[i])
    return C

        
def SVD(datasets,covariance_matrix, dimension, r): 
    """ calculate top r eigenvector of sample covariance matrix,number of nodes,dimension  and save top r eigenvector as pickle file"""
    eig_val, eig_vec = np.linalg.eig(covariance_matrix)
    Eig_vec_top_r = np.zeros((dimension,r))
    Eig_vec_top_r = eig_vec[:,0:r]

    with open("Data/Test/{}/Eig_vec_top.pickle".format(datasets), 'wb') as handle:
        pickle.dump(Eig_vec_top_r, handle)
                
    print('top {} repsent {} of full datasets'.format(r,np.sum(eig_val[0:r])/np.sum(eig_val)))
    return Eig_vec_top_r



def zeromean(data,dim,N):
    """Subtract each column with their mean """
    M=np.mean(data,axis=1).reshape(dim,1)
    M_matrix= np.tile(M,(1,N))
    return (data-M_matrix)

def load_MNIST_data():
    # Load mnist dataset
    mnist=pd.read_pickle("Data/Original/mnist_py3k.pkl.gz")
    train_data=mnist[0][0]
    train_label=mnist[0][1]
    test_data=mnist[2][0]
    test_label=mnist[2][1]
    
    data=np.array(train_data)
    TotalSamples=len(data)
    dataDimension=len(data[0])
    #np.random.shuffle(data)
    A0 = data.transpose()
    data0 = zeromean(A0,dataDimension,TotalSamples) # make data zero mean
    return data0,TotalSamples,dataDimension
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_CIFAR10_data():
    # Load cifar 10 dataset
    import pickle
    data=[]
    for i in range (1,6):
        f = 'Data/Original/cifar-10-batches-py/data_batch_{}'.format(i)
 
        train_dict=unpickle(f)
        train_data=train_dict[b'data']
        if (i == 1): 
            data=train_data
        else:  
            data = np.concatenate((data, train_data))
    

    data = data[:,:1024]
 
    TotalSamples = len(data)
    dataDimension = len(data[0])
    A0 = data.transpose()
    data = zeromean(A0,dataDimension,TotalSamples) # make data zero mean
    return data,TotalSamples,dataDimension

def load_lfw_people():
    # Load lfw dataset
    from sklearn.datasets import fetch_lfw_people
    lfw_dataset = fetch_lfw_people()
    TotalSamples, h, w = lfw_dataset.images.shape
    dataDimension=h*w
    data0 = lfw_dataset.data
    
    A0 = data0.transpose()
    data = zeromean(A0,dataDimension,TotalSamples) # make data zero mean
    return data,TotalSamples,dataDimension

def zeromean_ImageNet32_data():
    """Subtract each column with their mean for imagenet dataset"""
    import pickle
    file_size=5000
    dim=32*32*3
    number_of_nodes=200
    data=np.zeros((dim,file_size))
    
    for i in range (number_of_nodes):
        f = 'Data/Original/ImageNet32/data{}.pickle'.format(i)

        local_data=unpickle(f)
        data += local_data

    M=np.mean(data/number_of_nodes,axis=1).reshape(dim,1)
    M_matrix= np.tile(M,(1,file_size))
    
    for i in range (number_of_nodes):
        f = 'Data/Original/ImageNet32/data{}.pickle'.format(i)
        local_data=unpickle(f)
        with open('Data/Test/imagenet32/data{}.pickle'.format(i), 'wb') as handle:
                    pickle.dump(local_data-M_matrix, handle)
        print(i)
    
def SVD_ImageNet32_data(number_of_nodes,r):
    """ calculate top r eigenvector of sample covariance matrix,number of nodes,dimension and save top r eigenvector as pickle file for imagenet dataset"""
    import pickle
    file_size=5000
    data=np.zeros((1024,1024))
    for i in range (number_of_nodes):
        f = 'Data/Test/imagenet32/data{}.pickle'.format(i)
 
        local_data0=unpickle(f)
        local_data=local_data0[:1024,:]
        data += (1/(file_size))*np.dot(local_data,local_data.transpose())
        

    SVD("imagenet32",data, 1024, r)
       
        
        
        
        
        
        
        
        
        
        
        
        
