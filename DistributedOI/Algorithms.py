import numpy as np
from numpy import linalg as LA
from mpi4py import MPI
import pandas as pd
import pickle
import os
import time
import random
#error_type=angle/frobenius


class Experiment():
    def __init__(self,datasets,graphType,top_rank_r,number_of_nodes,node_i,iterations=100,T_consensus_init=1,T_consensus_max=50,Tc_inc=1,Straggler=False):
        self.datasets=datasets        # name of the datasets
        self.r=top_rank_r             # low rank r
        self.n=number_of_nodes        # n is the number of nodes
        self.node_i=node_i            # node index in the graph topology
        #self.X_init=X_init           # the initial value of X_init
        self.error_type='angle'       # define the type 
        self.itr=iterations           # number of iterations of orthogonal iterations
        self.tc_init=T_consensus_init # initial value of tc
        self.tc_max=T_consensus_max   # upper bound of tc
        self.tc_inc=Tc_inc            # tc=Tc_inc*t+T_consensus_init
        self.topology=graphType
        if Tc_inc==0:
            self.tc_type="const"
        else:
            self.tc_type="inc"
        self.Straggler=Straggler
        
    def load_local_data(self):#name of the file, id of the current node
        # Load the dataset
        data=pd.read_pickle("Data/Test/{}/data{}.pickle".format(self.datasets,self.node_i))
 
        self.dataDimension=data.shape[0]
        self.NumSamples=data.shape[1]

        self.covariance_matrix = (1/(self.NumSamples))*np.dot(data,data.transpose())
        

        X_init=pd.read_pickle('Data/Test/{}/X_init.pickle'.format(self.datasets))
        self.X_init=X_init[:,:self.r]

        top_r_eigenvec=pd.read_pickle('Data/Test/{}/Eig_vec_top.pickle'.format(self.datasets))
        self.top_r_eigenvec=top_r_eigenvec[:,:self.r]

        w=pd.read_pickle('Data/Test/{}/weight_{}.pickle'.format(self.datasets,self.topology))
        self.W=w[self.node_i,:]
        
        print("Successful load data ({},{})for node {}".format(self.dataDimension,self.NumSamples,self.node_i))
        print("local covariance:",self.covariance_matrix.shape)
        print("top_r_eigenvec:",self.top_r_eigenvec.shape)
        print("X_init:",self.X_init.shape)


    def Error(self,X,Y):
        if self.error_type=="angle":
            err=self.Angle_error(X,Y) 
        else:
            err=self.Frobenius_error(X,Y)
        return err 

    def Angle_error(self,X,Y): #dsistance between two subspaces
        X, R = LA.qr(X)
        Y, R = LA.qr(Y)
        k = X.shape[1]
        M = np.dot(X.transpose(),Y)
        u, s, v = LA.svd(M)
        sin_sq = 1 - s**2
        dist = np.sum(sin_sq)/k
        return dist


    def Frobenius_error(self,X,Y): #error given by Frobenius norm
        X=np.matmul(X,X.transpose())
        Y=np.matmul(Y,Y.transpose())
        error=np.linalg.norm(X-Y)
        return error

    def Get_local_result(self):
        return self.Q
    def Get_local_error(self):
        return self.error
    
    def ColumnDistributedOI(self):
        comm=MPI.COMM_WORLD 
        if self.tc_type=="inc":
            folder_name='{}_n{}r{}{}_{}{}'.format(self.datasets,self.n,self.r,self.topology,self.tc_type,int(self.tc_inc))
        else:
            folder_name='{}_n{}r{}{}_{}{}'.format(self.datasets,self.n,self.r,self.topology,self.tc_type,self.tc_max)
        
        if self.node_i==0 and not os.path.exists('result/{}'.format(folder_name)):
            os.mkdir('result/{}'.format(folder_name))
          

        self.load_local_data()

        T_c=[self.tc_init] # store tc values into an array
        T_consensus=self.tc_init # current T_consensus value
        self.number_comm=0
        Q=self.X_init

        err_curr=self.Error(self.top_r_eigenvec,Q)
        err =err_curr
        
        index=np.array([0])
        
        for t in range(self.itr):
            Q= np.dot(self.covariance_matrix, Q)

            if t>0:
                T_consensus=min(int(self.tc_init+self.tc_inc*t),self.tc_max)
                T_c.append(T_consensus)  

            for t_c in range(T_consensus):
                for i in range (self.n):
                    if (self.W[i]!=0) and (self.node_i!= i):
                        comm.send(Q*self.W[i],dest=i)
                        self.number_comm+=1
                        
                if self.Straggler:       
                    if self.node_i==0:
                        time.sleep(0.01)
                    #if random.uniform(0, 1)<0.05:
                        #sleep(0.01)
                        
                        
                update=np.zeros((self.dataDimension,self.r))
                for i in range (self.n):
                    if (self.W[i]!=0) and (self.node_i!= i):
                        update+=comm.recv(source=i)
                        
                Q=update+Q*self.W[self.node_i]

                err=np.append(err,err_curr)

            Q, R = LA.qr(Q)
            
            
            
            err_curr=self.Error(self.top_r_eigenvec,Q)
            err=np.append(err,err_curr)
            index=np.append(index,len(err)-1)
            
            if (t+1)%50==0:
                with open('result/{}/DOI_{}.pickle'.format(folder_name,self.node_i), 'wb') as handle:
                    pickle.dump(err, handle)
                if self.node_i==0:
                    with open('result/{}/TC_{}.pickle'.format(folder_name,self.node_i), 'wb') as handle:
                        pickle.dump(T_c, handle)
                    with open('result/{}/index.pickle'.format(folder_name), 'wb') as handle:
                        pickle.dump(index, handle)
        self.Q=Q
        self.error=err

        print("site: {},error: {},number of communication: {}".format(self.node_i,err_curr,self.number_comm))
        

    def consensus(self,local_val,TC):
        #w=pd.read_pickle('Data/Test/synthetic/weight.pickle')
        #self.W=w[self.node_i,:]
        
        comm=MPI.COMM_WORLD 
        Q=local_val
        
        for t_c in range(TC):
            for i in range (self.n):
                if (self.W[i]!=0) and (self.node_i!= i):
                    comm.send(Q*self.W[i],dest=i)
            update=0
            for i in range (self.n):
                if (self.W[i]!=0) and (self.node_i!= i):
                    update=update+comm.recv(source=i)
            Q=update+Q*self.W[self.node_i]

        return Q
