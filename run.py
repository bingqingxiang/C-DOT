#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from mpi4py import MPI
import time

# In[ ]:
from DistributedOI.Algorithms import Experiment


# In[ ]:

comm=MPI.COMM_WORLD 
my_rank = MPI.COMM_WORLD.Get_rank() # current node id
number_of_nodes=comm.Get_size()


# In[ ]:
r=5
#datasets="synthetic_n{}r{}g{}".format(number_of_nodes,r,7)
datasets="cifar{}".format(number_of_nodes)

start = time.time()

trail=Experiment(datasets=datasets,graphType='p025',top_rank_r=r,number_of_nodes=number_of_nodes,node_i=my_rank,iterations=400,T_consensus_init=1,T_consensus_max=50,Tc_inc=2)

trail.ColumnDistributedOI()

end = time.time()
print("runtime: ",end - start)


total_comm=trail.consensus(trail.number_comm,100)
print("total communication: ",total_comm)
#print("total communication: ",trail.number_comm)