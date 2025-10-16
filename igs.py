import numpy as np
import random
import json
import argparse


parser = argparse.ArgumentParser(description="Run experiment with configurable parameters.")
parser.add_argument('--workers', type=int, default=5,
                    choices=range(1,10),
                    # metavar='#workers',
                    help="Number of workers (default: 5)")
args = parser.parse_args()

# dast = args.dast
oracle_size = args.workers

dast = "Amazon"
print(f"Using dataset: {dast}")
print(f"Workers: {oracle_size}")

f=open('./Data/Amazon/Amazon.txt','r')
T=json.load(f)
f.close()

n=len(T)
m=n-1
sT=[0 for _ in range(n)]
reachable=[set() for _ in range(n)]

parent=[0 for i in range(n)]
for i in range(n):
    for j in T[i]:
        parent[j]=i
        
def dfs_reachable_set(u):
    reachable[u].add(u)
    for v in T[u]:
        dfs_reachable_set(v)
        reachable[u]=reachable[u]|reachable[v]

def dfs_subtree_size(x):
    sT[x]=1
    for i in range(len(T[x])):
        dfs_subtree_size(T[x][i])
        sT[x]=sT[x]+sT[T[x][i]]

def heavy_path_decomposition(T):
    count_heavy_path_node=0
    
    heavy_path_node_set=[]
    tree_node_to_heavy_path_node=[-1 for _ in range(len(T))]
    
    for i in range(n):
        if tree_node_to_heavy_path_node[i]==-1:
            tree_node_to_heavy_path_node[i]=count_heavy_path_node
            heavy_path_node_set.append([])
            u=i
            while True:
                tree_node_to_heavy_path_node[u]=count_heavy_path_node
                heavy_path_node_set[count_heavy_path_node].append(u)
                if len(T[u])>=1:
                    u=T[u][0]
                else:
                    break
            count_heavy_path_node=count_heavy_path_node+1
    return heavy_path_node_set,tree_node_to_heavy_path_node

dfs_reachable_set(0)
dfs_subtree_size(0)
for i in range(len(T)):
    T[i].sort(key=lambda x : sT[x],reverse=True)
heavy_path_node_set,tree_node_to_heavy_path_node=heavy_path_decomposition(T)

def get_noisy_answer(selected_node,target,num_votes):
    num_correct_answer=0
    num_wrong_answer=0
    true_answer=target in reachable[selected_node]
    worker_errate = [1-0.731, 1-0.26, 1-0.95, 1-0.95,1-0.26, 1-0.95,1-0.26,  1-0.95,1-0.26]
    # worker_errate = [0] * 9
    for i in range(num_votes):
        error_rate = worker_errate[i]
        if random.random()<error_rate:
            num_wrong_answer=num_wrong_answer+1
        else:        
            num_correct_answer=num_correct_answer+1
    #print('query on',selected_node,'get wrong answer',num_wrong_answer,'and correct answer',num_correct_answer)
    if num_wrong_answer>num_correct_answer:
        #print('WRONG ANSWER OCCURS')
        #print('return with answer:',not true_answer)
        return not true_answer
    else:
        #print('return with answer:',true_answer)
        return true_answer

def AIGS(target,num_votes):
    
    cost=0
    root=0
    progress=[]

    while True:
        current_path=heavy_path_node_set[tree_node_to_heavy_path_node[root]]
        
        l=0
        r=len(current_path)
        while (r-l>1):
            for i in range(num_votes-1):
                progress.append(current_path[l])
            
            cost=cost+num_votes
            m=(l+r)//2
            answer=get_noisy_answer(current_path[m],target,num_votes)
            if answer==True:
                l=m
            else:
                r=m
                
            progress.append(current_path[l])
        root=current_path[l]
        
        flag=False
        for i in range(1,len(T[root])):
            cost=cost+num_votes
            answer=get_noisy_answer(T[root][i],target,num_votes)
            if (answer==True):
                flag=True
                root=T[root][i]
                for i in range(num_votes):
                    progress.append(root)
                break
            else:
                for i in range(num_votes):
                    progress.append(root)
        if flag==False:
            # if root!=target:
                # print('Warning: Wrong Answer',root)
            return cost,root,progress
        
num_batch=1

task_id=0
task_cost=[]
task_result=[]
task_progress=[]
num_wrong_answer=0

for num_votes in [oracle_size]:
    for batch in range(num_batch):
        random.seed(batch)


        f=open('./Data/Amazon/AmazonItemsBatch'+str(batch)+'.txt','r')
        task_batch=json.load(f)
        f.close()    


        for target in task_batch:
            task_id=task_id+1

            cost,result,progress=AIGS(target,num_votes)

            if (target!=result):
                num_wrong_answer=num_wrong_answer+1

            task_cost.append(cost)
            task_result.append(result)
            task_progress.append(progress)
            # print(task_id,target)
            if task_id%10 == 0:
                print("total task:", task_id, ' number of wrong answers:',num_wrong_answer, ' Cost:', np.mean(task_cost))
            if task_id == 1000: break
        print("total task:", task_id, 'number of wrong answers:',num_wrong_answer, "accuracy: ", (task_id-num_wrong_answer)/task_id)
        print('Rounds:', np.mean(task_cost)/num_votes)