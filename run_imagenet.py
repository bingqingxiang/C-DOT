#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from mpi4py import MPI
import time

from DistributedOI.Algorithms_ImageNet import Experiment


# In[ ]:

comm=MPI.COMM_WORLD 
my_rank = MPI.COMM_WORLD.Get_rank() # current node id
number_of_nodes=comm.Get_size()


# In[ ]:

start = time.time()

trail=Experiment(datasets="imagenet32",graphType='p005',top_rank_r=5,number_of_nodes=number_of_nodes,node_i=my_rank,iterations=200,T_consensus_init=1,T_consensus_max=50,Tc_inc=2)

trail.ColumnDistributedOI()

end = time.time()
print("runtime: ",end - start)

total_comm=trail.consensus(trail.number_comm,100)
print("total communication: ",total_comm)
