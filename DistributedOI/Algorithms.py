import numpy as np
from numpy import linalg as LA
from mpi4py import MPI
import pandas as pd
import pickle
import os
import time
import random

class Experiment():
    """ This class construct the C-DOT and CA-DOT experiments """
    def __init__(self,datasets,graphType,top_rank_r,number_of_nodes,node_i,iterations=100,T_consensus_init=1,T_consensus_max=50,Tc_inc=1,Straggler=False):
        """ This is the _init_ function initialize all the parameters for the C-DOT experiment
        
            Keyword arguments:
            datasets -- The name of the dataset
            graphType -- The type the network topology
            top_rank_r -- The targeting dimension of the low rank subspace
            number_of_nodes -- Number of the nodes in the network
            node_i -- The id of the current node
            iterations --  Number of iterations for C-DOT and CA-DOT algorithm (default 100)
            T_consensus_init --  The initial value for the number of inner loop (default 1)
            T_consensus_max --  The maximum value for the number of inner loop (default 50)
            Tc_inc --  The increasing rate for the number of inner loop (default 1)
            Straggler --  Enable the straggler effect (default False)
            
        """
        self.datasets=datasets        # Name of the datasets
        self.topology=graphType       # Network topology
        self.r=top_rank_r             # Low rank r
        self.n=number_of_nodes        # Number of nodes n 
        self.node_i=node_i            # Node index in the network
        self.itr=iterations           # Number of iterations for C-DOT or CA-DOT
        self.tc_init=T_consensus_init # Initial value of tc
        self.tc_max=T_consensus_max   # Upper bound of tc
        self.tc_inc=Tc_inc            # Increasing rate of tc where tc=Tc_inc*t+T_consensus_init
        
        self.error_type='angle'       # Define the type 2-norm -- 'angle', frobenius norm -- 'frobenius' 
        
        # C-DOT -- "const", CA-DOT -- "inc"
        if Tc_inc==0:
            self.tc_type="const"
        else:
            self.tc_type="inc"
            
        self.Straggler=Straggler      # Enable straggler effect--  True , Disable traggler effect-- False
        
    def load_local_data(self):#name of the file, id of the current node
        """ Load local dataset 
            Calculate the local sample covariance matrix
            Load the initial value for Q^{init} 
            Load the true Top r low rank subspace Q calculated from SVD
            Load the weight matrix corresponding to the network
        """
        data=pd.read_pickle("Data/Test/{}/data{}.pickle".format(self.datasets,self.node_i))
 
        self.dataDimension=data.shape[0]
        self.NumSamples=data.shape[1]
        # Calculate the sample covariance
        self.covariance_matrix = (1/(self.NumSamples))*np.dot(data,data.transpose())
        
        X_init=pd.read_pickle('Data/Test/{}/X_init.pickle'.format(self.datasets))
        self.X_init=X_init[:,:self.r]

        top_r_eigenvec=pd.read_pickle('Data/Test/{}/Eig_vec_top.pickle'.format(self.datasets))
        self.top_r_eigenvec=top_r_eigenvec[:,:self.r]

        w=pd.read_pickle('Data/Test/{}/weight_{}.pickle'.format(self.datasets,self.topology))
        self.W=w[self.node_i,:]
        
        print("Successful load data with shape ({},{}) for node {}".format(self.dataDimension,self.NumSamples,self.node_i))
        print("Local sample covariance matrix has shape :",self.covariance_matrix.shape)
        print("Top r eigen vector has shape :",self.top_r_eigenvec.shape)
        print("Q^{init} has shape:",self.X_init.shape)


    def Error(self,X,Y):
        """ Choose the correct method to find the distance between two subspaces """
        if self.error_type=="angle":
            err=self.Angle_error(X,Y) 
        else:
            err=self.Frobenius_error(X,Y)
        return err 

    def Angle_error(self,X,Y): 
        """ Distance between two subspaces given by 2-norm"""
        X, R = LA.qr(X)
        Y, R = LA.qr(Y)
        k = X.shape[1]
        M = np.dot(X.transpose(),Y)
        u, s, v = LA.svd(M)
        sin_sq = 1 - s**2
        dist = np.sum(sin_sq)/k
        return dist


    def Frobenius_error(self,X,Y): 
        """ Distance between two subspaces given by Frobenius norm"""
        X=np.matmul(X,X.transpose())
        Y=np.matmul(Y,Y.transpose())
        error=np.linalg.norm(X-Y)
        return error

    def Get_local_result(self):
        """ Get the current estimation of Q"""
        return self.Q
    def Get_local_error(self):
        """ Get the current error betweent estimation of Q and true Q"""
        return self.error
    
    def ColumnDistributedOI(self):
        """ Execute the C-DOT/CA-DOT algorithm"""
        
        comm=MPI.COMM_WORLD 
        
        # Get the name of the result folder
        if self.tc_type=="inc":
            folder_name='{}_n{}r{}{}_{}{}'.format(self.datasets,self.n,self.r,self.topology,self.tc_type,int(self.tc_inc))
        else:
            folder_name='{}_n{}r{}{}_{}{}'.format(self.datasets,self.n,self.r,self.topology,self.tc_type,self.tc_max)
            
        # Create a result folder at node 0
        if self.node_i==0 and not os.path.exists('result/{}'.format(folder_name)):
            os.mkdir('result/{}'.format(folder_name))
          
        
        self.load_local_data()

        T_c=[self.tc_init]       # Store tc values into an array
        T_consensus=self.tc_init # Current number of inner loop (consensus averaging) T_consensus 
        self.number_comm=0       # Initialize the counter for number of point-to-point communication at current node
        Q=self.X_init            # Load the initial value for Q^{0}_{current node}

        err_curr=self.Error(self.top_r_eigenvec,Q)  # Calculate the error for the initial value
        err =err_curr            # Append the error into an error
        
        index=np.array([0])
        
        for t in range(self.itr): # Outter loop
            Q= np.dot(self.covariance_matrix, Q)

            if t>0:
                T_consensus=min(int(self.tc_init+self.tc_inc*t),self.tc_max) # Calculate current number of inner loop 
                T_c.append(T_consensus)  

            for t_c in range(T_consensus): # Inner loop
                for i in range (self.n):   # Send local estimation to all neighbors
                    if (self.W[i]!=0) and (self.node_i!= i):
                        comm.send(Q*self.W[i],dest=i)
                        self.number_comm+=1
                        
                if self.Straggler:         #Sleep for 0.01 second if Enabled straggler effect 
                    if self.node_i==0:
                        time.sleep(0.01)
                        
                update=np.zeros((self.dataDimension,self.r))
                for i in range (self.n):   # Receive estimations from all neighbors
                    if (self.W[i]!=0) and (self.node_i!= i):
                        update+=comm.recv(source=i)
                        
                Q=update+Q*self.W[self.node_i]  # Update the local estimation

                err=np.append(err,err_curr) 

            Q, R = LA.qr(Q) # QR factorization for local estimation 
            err_curr=self.Error(self.top_r_eigenvec,Q)
            err=np.append(err,err_curr)
            index=np.append(index,len(err)-1)
            
            # Save result after every 50 outter loop iterations
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
        """ Execute Consensus averaging algorithm
            Keyword arguments:
            local_val -- Local data scalar or matrix
            TC -- Number of iterations for consensus averaging
        """
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
