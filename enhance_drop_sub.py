"""
Copy each molecule and add 0 at the end of its CID
Do node dropping augmentation on each molecule and add 1 at the end of its CID
Do subraph augmentation on each molecule and add 2 at the end of its CID

TODO : do the same with subgraph (random walk)
"""

import os
import shutil
import random

source_data = "/home/infres/jalvarez-22/ALTEGRAD/Challenge/Public/data/"
target_data = "/home/infres/jalvarez-22/ALTEGRAD/Challenge/Public/data_enhanced/"

def node_drop(f_source,f_target,p=0.8) :
    """
    from a source file compute node drop in target file
    keep the proportion p of nodes
    """
    lines=f_source.readlines()
    index=lines.index("idx to identifier:\n") 
    nb_nodes=len(lines)-index-1
    #construct list of node to keep
    keep=list(range(nb_nodes))
    while len(keep)>int(p*nb_nodes) :
        keep.pop(random.randint(0,len(keep)-1))
    # construct mapping from old idx to new idx
    mapping={}
    k=0
    for idx in range(nb_nodes) :
        if k<int(p*nb_nodes) and idx==keep[k] :
            mapping[str(idx)]=str(k)
            k+=1
        else :
            mapping[str(idx)]=None

    if len(keep)==0 :
        # We copy the original file
        for line in lines :
            f_target.write(line)
        return
    
    #write edges
    f_target.write("edgelist:\n")
    for line in lines[1:index-1] :
        x1,x2=line[:-1].split(" ")
        y1,y2=mapping[x1],mapping[x2]
        if y1 is not None and y2 is not None :
            f_target.write(y1+' '+y2+'\n')

    f_target.write("\n")
    #write nodes
    f_target.write("idx to identifier:\n")
    for line in lines[index+1:] :
        x,identifier=line[:-1].split(" ")
        y=mapping[x]
        if y is not None:
            f_target.write(y+' '+identifier+'\n')

def get_neighbors(node, edges):
    ''' Get the neighbors of a node from a list of edges '''

    neighbors = []
    for edge in edges:
        if node in edge:
            neighbors.append(edge[0] if edge[0] != node else edge[1])

    # remove duplicates
    neighbors = list(set(neighbors))
    return neighbors

def subgraph_random_walk(f_source,f_target,rate=0.8):
    """
    from a source file compute subgraph random walk in target file
    rate is the proportion of nodes to keep
    """

    lines=f_source.readlines()
    index=lines.index("idx to identifier:\n") 
    nb_nodes=len(lines)-index-1

    nb_subgraph = int(nb_nodes*rate)

    # list of edges
    edges = []
    for line in lines[1:index-1] :
        x1,x2=line[:-1].split(" ")
        edges.append((x1,x2))
    # list of nodes
    nodes = []
    for line in lines[index+1:] :
        x,identifier=line[:-1].split(" ")
        nodes.append(x)

    # random walk
    current_node = random.choice(nodes)
    subgraph = [current_node]
    for i in range(nb_subgraph):
        neighbors = get_neighbors(current_node, edges)
        if len(neighbors) == 0:
            break
        current_node = random.choice(neighbors)
        subgraph.append(current_node)

    #write edges
    f_target.write("edgelist:\n")
    for line in lines[1:index-1] :
        x1,x2=line[:-1].split(" ")
        if x1 in subgraph and x2 in subgraph:
            f_target.write(x1+' '+x2+'\n')

    f_target.write("\n")
    #write nodes
    f_target.write("idx to identifier:\n")
    for line in lines[index+1:] :
        x,identifier=line[:-1].split(" ")
        if x in subgraph:
            f_target.write(x+' '+identifier+'\n')


#copy each original molecule and add 0 at the end of the ID
#do node dropping augmentation and add 1 at the end of the ID

try : os.mkdir(target_data)
except : pass
try : os.mkdir(target_data+"raw/")
except : pass

print("\nCopy single files")
shutil.copyfile(source_data+"test_text.txt", target_data+"test_text.txt")
shutil.copyfile(source_data+"token_embedding_dict.npy", target_data+"token_embedding_dict.npy")

with open(source_data+"test_cids.txt", 'r') as f_source :
    with open(target_data+"test_cids.txt", 'w+') as f_target :
        for l in f_source.readlines() :
            f_target.write(l[:-1]+"0\n")

with open(source_data+"train.tsv", 'r') as f_source :
    with open(target_data+"train.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"0\t"+text)

with open(source_data+"train.tsv", 'r') as f_source :
    with open(target_data+"train_drop.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"1\t"+text)

with open(source_data+"train.tsv", 'r') as f_source :
    with open(target_data+"train_subgraph.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"2\t"+text)
           

with open(source_data+"val.tsv", 'r') as f_source :
    with open(target_data+"val.tsv", 'w+') as f_target :
        for l in f_source.readlines() :
            cid,text=l.split('\t')
            f_target.write(cid+"0\t"+text)

print("\nCopy raw files")
c = 0
for f in os.listdir(source_data+"raw/") :
    c += 1
    cid=f.split('.')[0]
    shutil.copyfile(source_data+"raw/"+f, target_data+"raw/original/"+cid+"0.graph")
    print("\r" + str(c) + " of " + str(len(os.listdir(source_data+"raw/"))), end='')



print("\nNode dropping")
c = 0
for f in os.listdir(source_data+"raw/") :
    c += 1
    if f!= ".DS_Store" :
        cid=f.split('.')[0]
        with open(source_data+"raw/"+f, 'r') as f_source :
            with open(target_data+"raw/drop/"+cid+"1.graph", 'w+') as f_target :
                node_drop(f_source,f_target)
        print("\r" + str(c) + " of " + str(len(os.listdir(source_data+"raw/"))), end='')

print("\nSubgraph random walk")
c = 0
for f in os.listdir(source_data+"raw/") :
    c += 1
    if f!= ".DS_Store" :
        cid=f.split('.')[0]
        with open(source_data+"raw/"+f, 'r') as f_source :
            with open(target_data+"raw/subgraph/"+cid+"2.graph", 'w+') as f_target :
                subgraph_random_walk(f_source,f_target)
        print("\r" + str(c) + " of " + str(len(os.listdir(source_data+"raw/"))), end='')
