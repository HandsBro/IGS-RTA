import numpy as np
import random
from scipy.special import erf, expit as logistic
from scipy.optimize import minimize
from dataclasses import dataclass, field
from itertools import combinations
from typing import List
import numpy as np
import random
import json
import time
import sys
import math
import argparse


parser = argparse.ArgumentParser(description="Run experiment with configurable parameters.")

parser.add_argument('--dast', type=str, default='Amazon',
                    choices=['Amazon', 'Imagenet'],
                    help="Dataset name: 'Amazon' or 'Imagenet' (default: Amazon)")
parser.add_argument('--threshold', type=int, default=99,
                    choices=[60, 70, 80, 90, 99],
                    # metavar='termination_threshold',
                    help="Threshold value (default: 99).")
parser.add_argument('--workers', type=int, default=3,
                    choices=[3, 5, 7, 9],
                    # metavar='#workers',
                    help="Number of workers (default: 3)")
args = parser.parse_args()

dast = args.dast
threshold = args.threshold
workers = args.workers

# dast = "Amazon"
print(f"Using dataset: {dast}")
print(f"Threshold: {threshold}")
print(f"Workers: {workers}")

#dataset
if dast == "Imagenet": f=open('../Data/Imagenet/ImagenetT.txt','r')
else: f=open('../Data/Amazon/Amazon.txt','r')
T=json.load(f)
f.close()

n=len(T)
m=n-1
reachable=[set() for _ in range(n)]
pT=[0 for _ in range(n)]

#dataset Constants
num_batch=1 #total 10
threshold=99

# EM Constants
workers = 7
worker_pool = [-1,1,3,-1,1,-1,1]
# worker_pool = [-1,1,3,-1,3,-1,3,-1,3]
# worker_pool = [-3,-3,-3,-3,-3,3,3,3,3] #negative test.
tasks = 0
ALPHA_PRIOR_MEAN = 1
BETA_PRIOR_MEAN = 1
CONVERGENCE_THRESHOLD = 1e-5
EM_BATCH_SIZE = workers * 100
# alphas_gt = np.random.normal(loc=1, scale=1, size=workers)
#============================ normal adversarial
# alphas_gt[0] = -1 #adversarial
# alphas_gt[1] = 1 #normal
# alphas_gt[2] = 3 #right
# alphas_gt[3] = 1 #normal
# alphas_gt[4] = 3 #right
# alphas_gt[5] = 1 #normal
# alphas_gt[6] = 3 #right
# alphas_gt[7] = 1 #normal
# alphas_gt[8] = 3 #right
#============================ more adversarial
alphas_gt = worker_pool[0:workers:1]
# alphas_gt[0] = -1 #adversarial
# alphas_gt[1] = 1 #normal
# alphas_gt[2] = 3 #right
# alphas_gt[3] = -1 #normal
# alphas_gt[4] = 3 #right
# alphas_gt[5] = -1 #normal
# alphas_gt[6] = 3 #right
# alphas_gt[7] = -1 #normal
# alphas_gt[8] = 3 #right
#============================ all adversarial
# alphas_gt[0] = -1 #adversarial
# alphas_gt[1] = -1 #normal
# alphas_gt[2] = -3 #right

normal_draws = np.random.normal(loc=1, scale=1, size=n) 
betas_gt = np.exp(normal_draws)


alphas_es = [1] * workers
betas_es = [1] * n

beta_inference_id = set()

# print(f"OIGS Alpha MAE: {np.mean(np.abs(np.array(alphas_gt) - np.array(alphas_es)))}")
# print(f"OIGS Beta MAE: {np.mean(np.abs(np.array(betas_gt) - np.array(betas_es)))}")

@dataclass
class Label:
    imageIdx: int
    labelerId: int
    label: int

@dataclass
class Dataset:
    Imageid = []
    numLabels: int = 0
    numLabelers: int = 0
    numImages: int = 0 
    priorZ1: int = 0
    priorAlpha: np.ndarray = field(default_factory=lambda: np.array([]))
    priorBeta: np.ndarray = field(default_factory=lambda: np.array([]))
    probZ1: np.ndarray = field(default_factory=lambda: np.array([]))
    probZ0: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    beta: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: List[Label] = field(default_factory=list)

    def get_last_task_labels(self):
        if not self.labels:
            return []
        
        last_task_idx = (self.numImages - 1) % EM_BATCH_SIZE
        
        last_task_labels = [
            label.label
            for label in self.labels 
            if label.imageIdx == last_task_idx
        ]

        return last_task_labels

def init(data: Dataset):
    data.Imageid = []
    data.numLabels = 0 
    data.numLabelers = workers
    data.numImages = 0
    data.priorZ1 = 0.5
    data.priorAlpha = np.full(data.numLabelers, ALPHA_PRIOR_MEAN)
    data.priorBeta = np.full(data.numImages, BETA_PRIOR_MEAN)
    # data.probZ1 = np.full(data.numImages, data.priorZ1)
    # data.probZ0 = 1 - data.probZ1
    data.alpha = np.zeros(data.numLabelers)
    data.beta = np.zeros(data.numImages)

def update_data(data: Dataset, node, true_answer):
    global tasks, pT
    tasks += 1
    data.priorZ1 = 0.5
    if (tasks <= EM_BATCH_SIZE): 
        data.Imageid.append(node) #query node list. should be rounded
        # data.priorZ1 = np.append(data.priorZ1, pT[node]) #should be rounded
    else:
        data.Imageid[data.numImages] = node
        # data.priorZ1[data.numImages] = pT[node]

    data.numLabelers = workers
    data.numLabels += workers  #total label
    
    data.priorAlpha = np.full(data.numLabelers, ALPHA_PRIOR_MEAN)
    data.priorBeta = np.full(min(tasks, EM_BATCH_SIZE), BETA_PRIOR_MEAN)

    # data.priorAlpha = np.array(alphas_es)
    # beta_prior = []
    # for id in data.Imageid: beta_prior.append(betas_es[id])
    # data.priorBeta = np.array(beta_prior)

    data.probZ1 = np.full(min(tasks, EM_BATCH_SIZE), data.priorZ1)
    data.probZ0 = 1 - data.probZ1

    data.alpha = np.zeros(data.numLabelers)
    data.beta = np.zeros(min(tasks, EM_BATCH_SIZE))  

    for i in range(workers):
        accuracy = 1 / (1+np.exp(- alphas_gt[i]*betas_gt[node])) ## generated by true parameters
        if(random.uniform(0, 1) <= accuracy): ##correct answer
            label = Label(
                imageIdx=data.numImages,
                labelerId=i,
                label = true_answer 
            )
        else:  ##wrong answer
            label = Label(
                imageIdx=data.numImages,
                labelerId=i,
                label = not true_answer
            )
        # print(label.label)
        if (tasks <= EM_BATCH_SIZE): 
            data.labels.append(label)
        else: 
            data.labels[data.numImages*workers + i] = label
        # print(tasks, data.numImages, len(data.labels))
    
    data.numImages = (data.numImages + 1) % EM_BATCH_SIZE 
    
def pack_parameters(x: np.ndarray, data: Dataset):
    x[:data.numLabelers] = data.alpha
    x[data.numLabelers:] = data.beta

def unpack_parameters(x: np.ndarray, data: Dataset):
    data.alpha = x[:data.numLabelers].copy() 
    data.beta = x[data.numLabelers:].copy()

def compute_log_probability(l: int, z: int, alpha: float, beta: float):
    safe_beta = np.clip(beta, -50, 50) 
    
    exponent = np.exp(safe_beta) * alpha
    
    exponent = np.clip(exponent, -50, 50)
    
    if z == l:
        return -np.log1p(np.exp(-exponent)) 
    else:
        return -np.log1p(np.exp(exponent))

def expectation_step(data: Dataset):
    global tasks

    log_prob_z1 = np.log(data.priorZ1) 
    log_prob_z0 = np.log(1 - data.priorZ1)

    evidence_z1 = np.zeros(min(tasks, EM_BATCH_SIZE))
    evidence_z0 = np.zeros(min(tasks, EM_BATCH_SIZE))
    
    for label in data.labels:
        i = label.labelerId  
        j = label.imageIdx   
        lij = label.label    

        evidence_z1[j] += compute_log_probability(lij, 1, data.alpha[i], data.beta[j])
        evidence_z0[j] += compute_log_probability(lij, 0, data.alpha[i], data.beta[j])
    
    log_prob_z1 += evidence_z1
    log_prob_z0 += evidence_z0
    
    max_log = np.maximum(log_prob_z1, log_prob_z0)  
    prob_z1 = np.exp(log_prob_z1 - max_log)         
    prob_z0 = np.exp(log_prob_z0 - max_log)
    total = prob_z1 + prob_z0                      

    data.probZ1 = prob_z1 / total
    data.probZ0 = prob_z0 / total

def compute_Q_function(data: Dataset):
    Q = 0.0
    
    eps = 1e-300  
    Q += np.sum(data.probZ1 * np.log(data.priorZ1 + eps) + 
         data.probZ0 * np.log(1 - data.priorZ1 + eps))
    

    for label in data.labels:
        i = label.labelerId
        j = label.imageIdx

        safe_beta = np.clip(data.beta[j], -50, 50)
        exponent = np.exp(safe_beta) * data.alpha[i]
        exponent = np.clip(exponent, -50, 50)

        log_sigma = -np.log1p(np.exp(-exponent))
        log_one_minus_sigma = -np.log1p(np.exp(exponent))
        
        Q += (data.probZ1[j] * (label.label * log_sigma + (1 - label.label) * log_one_minus_sigma) +
              data.probZ0[j] * ((1 - label.label) * log_sigma + label.label * log_one_minus_sigma))
    
    alpha_diff = data.alpha - data.priorAlpha
    beta_diff = data.beta - data.priorBeta
    
    erf_alpha = np.clip(erf(alpha_diff), a_min=eps, a_max=None)
    erf_beta = np.clip(erf(beta_diff), a_min=eps, a_max=None)
    
    Q += np.sum(np.log(erf_alpha)) + np.sum(np.log(erf_beta))
    
    return Q

def gradient_Q(data: Dataset):
    dQdAlpha = -(data.alpha - data.priorAlpha)
    dQdBeta = -(data.beta - data.priorBeta)
    
    for label in data.labels:
        i = label.labelerId
        j = label.imageIdx

        safe_beta = np.clip(data.beta[j], -50, 50)
        
        exponent = np.exp(safe_beta) * data.alpha[i]
        exponent = np.clip(exponent, -50, 50) 
        
        sigma = logistic(exponent)
        
        term = (data.probZ1[j] * (label.label - sigma) + 
                data.probZ0[j] * (1 - label.label - sigma))
        

        dQdAlpha[i] += term * np.exp(safe_beta)
        dQdBeta[j] += term * data.alpha[i] * np.exp(safe_beta)
    
    return dQdAlpha, dQdBeta

def maximization_step(data: Dataset):

    def objective(x: np.ndarray):
        unpack_parameters(x, data)
        return -compute_Q_function(data) 
    
    def gradient(x: np.ndarray):
        unpack_parameters(x, data)
        dQdAlpha, dQdBeta = gradient_Q(data)
        return np.concatenate([-dQdAlpha, -dQdBeta])  
    
    x0 = np.concatenate([data.alpha, data.beta])
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 50, 'gtol': 1e-3}
    )

    unpack_parameters(result.x, data)

def expectation_maximization(data: Dataset):
    np.random.seed(int(time.time()))
    
    data.alpha = data.priorAlpha.copy()
    data.beta = data.priorBeta.copy()
    
    expectation_step(data)
    current_Q = compute_Q_function(data)
    
    while True:
        last_Q = current_Q
        expectation_step(data)
        current_Q = compute_Q_function(data)
        
        maximization_step(data)
        current_Q = compute_Q_function(data)
        
        if abs((current_Q - last_Q) / last_Q) <= CONVERGENCE_THRESHOLD:
            break

def output_results(data: Dataset):
    print(f"labelers: {data.numLabelers}, tasks: {data.numImages}.")
    print("\nFinal Parameters:")
    for i, alpha in enumerate(data.alpha):
        print(f" Alpha[{i}] = {alpha:.4f}; GT: {alphas_gt[i]}")

def calculate_answer_error_rate(error_rates, votes, final_answer, prior=0.5):

    p_votes_given_T = 1.0
    p_votes_given_not_T = 1.0
    
    for e, vote in zip(error_rates, votes):
        if final_answer == 1:
            p_votes_given_T *= (1 - e) if vote == 1 else e
            p_votes_given_not_T *= e if vote == 1 else (1 - e)
        else:
            p_votes_given_T *= (1 - e) if vote == 0 else e
            p_votes_given_not_T *= e if vote == 0 else (1 - e)

    prior_T = prior
    prior_not_T = 1-prior
    p_votes = p_votes_given_T * prior_T + p_votes_given_not_T * prior_not_T
    p_error = (p_votes_given_not_T * prior_not_T) / p_votes
    
    return p_error

def target_probability_increase(selected_node):
    pTi=pT[selected_node]
    err=e[selected_node]
    # balance_rate=max(pTi/(1-pTi),(1-pTi)/pTi)
    # return (1-2*err)*(1/(balance_rate+err/(1-err))-1/(balance_rate+(1-err)/err))
    # return pTi*(1-2*err)*(1/(pTi/(1-pTi)+err/(1-err))-1/(pTi/(1-pTi)+(1-err)/err)) + (1-pTi)*(1-2*err)*(1/((1-pTi)/pTi+err/(1-err))-1/((1-pTi)/pTi+(1-err)/err))
    return min((1-2*err)*(1/(pTi/(1-pTi)+err/(1-err))-1/(pTi/(1-pTi)+(1-err)/err)), (1-2*err)*(1/((1-pTi)/pTi+err/(1-err))-1/((1-pTi)/pTi+(1-err)/err)))

def select_query_node():
    selected_node=1
    expected_increase=target_probability_increase(selected_node)
    for i in range(2,n):
        tmp=target_probability_increase(i)
        if (tmp>expected_increase):
            expected_increase=tmp
            selected_node=i
    # print(f"Node:{selected_node}, X:{pT[selected_node]}, Y:{e[selected_node]}, E:{expected_increase}")
    pTi=pT[selected_node]
    err=e[selected_node]
    # print((1-2*err)*(1/(pTi/(1-pTi)+err/(1-err))-1/(pTi/(1-pTi)+(1-err)/err)), (1-2*err)*(1/((1-pTi)/pTi+err/(1-err))-1/((1-pTi)/pTi+(1-err)/err)))
    return selected_node

def majority_vote_probability(acc):

    k = len(acc)  
    threshold = math.ceil(k / 2) 
    
    total_probability = 0.0
    for m in range(threshold, k + 1):
        for correct_workers in combinations(range(k), m):
            prob = 1.0
            for i in range(k):
                if i in correct_workers:
                    prob *= acc[i]  # correct
                else:
                    prob *= (1 - acc[i])  # wrong
            total_probability += prob
    return total_probability

def select_node_entropy():
    increase = 0
    answer = -1
    e_entropy = []
    for i in range(n):
        acc_list = []
        for j in range(workers):
            exponent = -alphas_es[j] * betas_es[i]
            
            exponent = np.clip(exponent, -50, 50)
            
            acc_list.append(1 / (1 + np.exp(exponent)))

        correct = majority_vote_probability(acc_list)
        # print(correct)
        e_entropy.append(1-correct)


    for i in range(n):
        Y = e_entropy[i]
        X = pT[i]
        # if(X==0 and Y==0): print(f"{i}: e: {Y}; pT: {X}")
        Z = (X-2*X*Y+Y)*np.log(X-2*X*Y+Y) + (1+2*X*Y-X-Y)*np.log(1+2*X*Y-X-Y) - Y*np.log(Y) - (1-Y)*np.log(1-Y)
        # print(i, Z)
        if Z < increase:
            increase=Z
            answer=i
    return answer

def select_query_node_middle():
    selected = 0
    gap = 1
    for i in range(n):
        if (abs(pT[i]-0.5) < gap):
            gap = abs(pT[i]-0.5)
            selected = i
    # print(f"gap: {gap}; pT closest to 0.5: {pT[selected]}; max individual: {max(p)}")
    return selected

def dfs_subtree_weight(u,pT):
    pT[u]=p[u]
    for v in T[u]:
        dfs_subtree_weight(v,pT)
        pT[u]=pT[u]+pT[v]

def dfs_reachable_set(u):
    reachable[u].add(u)
    for v in T[u]:
        dfs_reachable_set(v)
        reachable[u]=reachable[u]|reachable[v]

dfs_reachable_set(0)

def get_noisy_answer(true_answer,num_votes,error_rate):
    num_correct_answer=0
    num_wrong_answer=0
    for i in range(num_votes):
        if random.random()<error_rate:
            num_wrong_answer=num_wrong_answer+1
        else:        
            num_correct_answer=num_correct_answer+1
    if num_wrong_answer>num_correct_answer:
        return not true_answer
    else:
        return true_answer

def normalize(p):
    sum_p=sum(p)
    for i in range(n):
        p[i]=p[i]/sum_p

def update_probability(selected_node,answer):
    err=e[selected_node]
    for i in range(n):
        if (answer == (i in reachable[selected_node])):
            p[i]=p[i]*(1-err)
        else:
            p[i]=p[i]*err
    normalize(p)
      
def update_probability_new(selected_node,answer,prob):
    err=prob
    for i in range(n):
        if (answer == (i in reachable[selected_node])):
            p[i]=p[i]*(1-err)
        else:
            # print()
            p[i]=p[i]*err
    normalize(p)

def update_probability_em(data, node):
    idx = (data.numImages - 1) % EM_BATCH_SIZE
    votes = data.get_last_task_labels()
    # print(votes)
    beta = betas_es[idx]
    errors = []
    for i in range(workers):
        errors.append(1 - 1 / (1+np.exp( - alphas_es[i]*beta))) 
    
    if (data.probZ1[idx]>0.5): final_ans = 1
    else: final_ans = 0
    # print(errors)

    pe = calculate_answer_error_rate(errors, votes, final_ans)
    pe = max(0.05, pe)
    # print(f'pe: {pe}')
    update_probability_new(node, final_ans, pe)

def update_error_rates(data):
    global beta_inference_id
    nodes = data.Imageid
    esbetas = data.beta
    betameans = {}

    for i in range(len(nodes)):
    # for node in nodes:
        node = nodes[i]
        if node in betameans.keys():
            betameans[node][0]+=esbetas[i]
            betameans[node][1]+=1
        else:
            betameans[node] = [esbetas[i], 1]
    
    for i in betameans.keys():
        beta_inference_id.add(i)
        new_beta = betameans[i][0]/ betameans[i][1]
        betas_es[i] = new_beta
    
    for i in range(workers):
        alphas_es[i] = data.alpha[i]
        # print(f"{i}: {alphas_es[i]}")

def NIGS(data, target, threshold):
    global p
    
    f=open(f'../Data/{dast}/{dast}NodeProbability.txt','r')
    p=json.load(f)
    f.close()

    max_p=max(p)

    cost=0
    round = 0
    progress=[]
    while max_p<=threshold:
        round += 1
        cost=cost+1
        dfs_subtree_weight(0,pT)
        selected_node = select_node_entropy() 
        update_data(data, selected_node, target in reachable[selected_node])
        expectation_maximization(data)
        update_error_rates(data)
        update_probability_em(data,selected_node)

        max_p=max(p)
    output_results(data)

    all_set = []
    for i in range(len(p)):
        if p[i] > threshold:
            all_set.append(i)

    return cost,p.index(max_p),all_set


if __name__ == "__main__":

    data = Dataset()
    init(data)

    print(f"dataset:{dast}, threshold:{threshold}.")

    for batch in range(num_batch):
        random.seed(batch)

        f=open(f'../Data/{dast}/{dast}ItemsBatch'+str(batch)+'.txt','r')
        task_batch=json.load(f)
        f.close()

        task_id=0
        task_cost=[]
        task_result=[]
        task_progress=[]
        num_wrong_answer = 0

        for target in task_batch:
            
            task_id=task_id+1

            cost,result,progress=NIGS(data, target, threshold/100)
            task_cost.append(cost)
            task_result.append(result)
            task_progress.append(progress)
            print(task_id,target)

            if(target != result): num_wrong_answer+=1

            print("total task:", task_id, ' number of wrong answers:', num_wrong_answer, ' asking rounds:', np.mean(task_cost), " costs:", np.mean(task_cost)*workers)
            if task_id == 1000: break

print("=========")
print("total task:", task_id, ' number of wrong answers:', num_wrong_answer, ' asking rounds:', np.mean(task_cost), " costs:", np.mean(task_cost)*workers)