from environment import *
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
import copy
import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from buffer import ReplayBuffer
from utils import save_snapshot, recover_snapshot, load_model
from schedule import LinearSchedule

from datetime import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys


###############################################
########### 1.Agent for Q-Learninig ###########
###############################################
class QTable:
    def __init__(self, num_states, num_actions, gamma=0.99, alpha_const=None, d=1):
        self.gamma = gamma
        self.visit_count = np.zeros(shape=(num_states, num_actions))
        self.Q = np.zeros(shape=(num_states, num_actions))
        self.alpha_const = alpha_const
        self.d = d

    def update(self, state, action, reward, next_state, done):      
        alpha = 1. / (self.visit_count[state, action] + 1) ** self.d
        if self.alpha_const is not None:
            alpha = self.alpha_const
        self.visit_count[state, action] += 1
        
        self.Q[state,action] = self.Q[state,action] + alpha*(reward +self.gamma*np.max(self.Q[next_state,:])  - self.Q[state,action] )
        if done: # To take terminal state into account - do not modify here!
            TQ = reward

    # Greedy policy - we will use this when testing
    def greedy(self, state):
        return np.argmax(self.Q[state])

    @property
    def value_ftn(self):
        return np.max(self.Q, axis=1)

def Qlearning(env, rollout_len = 1000000, alpha_const=None, d=0.50001):
    agent = QTable(num_states=env.num_category**env.num_gate, num_actions=env.num_gate, alpha_const=alpha_const, d=d)
    s = env.reset()
    epsilon = 0.5
    for t in tqdm(range(rollout_len + 1)):
        if np.random.rand(1) < epsilon:
            a = random.randint(0,env.num_gate-1) 
        else:
            max_val = np.max(agent.Q[s,:]) 
            idx = np.where(agent.Q[s,:]== max_val)[0].tolist() 
            a = random.sample(idx,1)[0]
            #a = np.argmax(agent.Q[s,:])      # 여기서도 random sampling이 필요.

        # Perform action
        next_s, r, done, _ = env.step(action=a)

        # Update our Q table
        agent.update(state=s, action=a, reward=r, next_state=next_s, done=done) # Transition to terminal state
        
        # State transition
        s = next_s
        if done == True:
            s = env.reset()

    return agent

###############################################
########### 2.Agent for double DQN  ###########
###############################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('current device =', device)

class Critic(nn.Module):
    def __init__(self, state_dim, num_action, hidden_size1, hidden_size2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_action)


    def forward(self, state):
        # given a state s, the network returns a vector Q(s,) of length |A|
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class DQNAgent:
    def __init__(self, obs_dim, num_act, hidden1, hidden2):
        self.obs_dim = obs_dim
        self.num_act = num_act
        # networks
        self.critic = Critic(obs_dim, num_act, hidden1, hidden2).to(device)
                
    def act(self, state, epsilon=0.0):
        # simple implementation of \epsilon-greedy method
        # TODO : Complete epsilon-greedy action selection
        # Hint : np.randon.rand() will generate random number in [0,1]
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_act)
        else:            # greedy selection
            self.critic.eval()
            s = torch.Tensor(state).view(1, self.obs_dim).to(device)
            q = self.critic(s)
            return np.argmax(q.cpu().detach().numpy())

def update(agent, replay_buf, gamma, critic_optim, target_critic, tau, batch_size):
    # agent : agent with networks to be trained
    # replay_buf : replay buf from which we sample a batch
    # actor_optim / critic_optim : torch optimizers
    # tau : parameter for soft target update
    
    agent.critic.train()

    batch = replay_buf.sample_batch(batch_size)

    # unroll batch
    with torch.no_grad():
        observations = torch.Tensor(batch['state']).to(device)
        actions = torch.tensor(batch['action'], dtype=torch.long).to(device)
        rewards = torch.Tensor(batch['reward']).to(device)
        next_observations = torch.Tensor(batch['next_state']).to(device)
        terminals = torch.Tensor(batch['done']).to(device)

        mask = 1.0 - terminals
        ### double DQN? ###
        a_inner = torch.unsqueeze(torch.max(agent.critic(next_observations), 1)[1], 1).detach()
        next_q_double = target_critic(observations).gather(1, a_inner)
        next_q_double = mask * next_q_double
        ###################
        # next_q = torch.unsqueeze(target_critic(next_observations).max(1)[0], 1)
        # next_q = mask * next_q
        
        # TODO : Build Bellman target for Q-update
        target = rewards + gamma * next_q_double 
        # target = rewards + gamma * next_q 

    out = agent.critic(observations).gather(1, actions)

    loss_ftn = MSELoss()
    loss = loss_ftn(out, target)

    critic_optim.zero_grad()
    loss.backward()
    critic_optim.step()
        
    # soft target update (both actor & critic network)
    for p, targ_p in zip(agent.critic.parameters(), target_critic.parameters()):
        targ_p.data.copy_((1. - tau) * targ_p + tau * p)
        
    return

def evaluate(agent, env, num_episodes=5):

    sum_scores = 0.
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        score = 0.
        
        while not done:
            action = agent.act(obs)
            obs, rew, done, _ = env.step(action)
            score += rew
        sum_scores += score
    avg_score = sum_scores / num_episodes
    
    return avg_score

def train(agent, env, gamma, 
          lr, tau,
          ep_len, num_updates, batch_size,
          init_buffer=5000, buffer_size=100000,
          start_train=2000, train_interval=50,
          eval_interval=2000, snapshot_interval=10000,
          path=None):
    
    target_critic = copy.deepcopy(agent.critic)
    
    # environment for evaluation
    test_env = copy.deepcopy(env)
    
    # freeze target network
    for p in target_critic.parameters():
        p.requires_grad_(False)

    critic_optim = Adam(agent.critic.parameters(), lr=lr)

    if path is not None:
        recover_snapshot(path, agent.critic,
                         target_critic, critic_optim,
                         device=device
                        )
    # load snapshot
    
    obs_dim = env.num_state
    num_act = env.num_gate
    
    replay_buf = ReplayBuffer(obs_dim, buffer_size)
    
    max_epsilon = 1.
    min_epsilon = 0.02
    exploration_schedule = LinearSchedule(begin_t=start_train,
                                          end_t=num_updates,
                                          begin_value=max_epsilon,
                                          end_value=min_epsilon
                                         )
    save_path = './snapshots/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./learning_curves/', exist_ok=True)
    log_file = open('./learning_curves/res.csv',
                    'w',
                    encoding='utf-8',
                    newline=''
                   )
    logger = csv.writer(log_file)
    
    # main loop
    obs = env.reset()
    done = False
    step_count = 0
    
    for t in range(num_updates + 1):
        if t < init_buffer:
            # perform random action until we collect sufficiently many samples
            # this is for exploration purpose
            action = random.randint(0,env.num_gate-1) 
        else:
            # executes epsilon-greedy action
            epsilon = exploration_schedule(t)
            action = agent.act(obs, epsilon=epsilon)
            
        next_obs, rew, done, _ = env.step(action)
        step_count += 1
        if step_count == ep_len:
            # if the next_state is not terminal but done is set to True by gym env wrapper
            done = False
            
        replay_buf.append(obs, action, next_obs, rew, done)
        obs = next_obs
        
        if done == True or step_count == ep_len:
            # reset environment if current environment reaches a terminal state 
            # or step count reaches predefined length
            obs = env.reset()
            done = False
            step_count = 0
            # score = evaluate(agent, env)
            # print('[iteration {}] evaluation score : {}'.format(t, score))
        
        if t % eval_interval == 0:
            avg_score = evaluate(agent, test_env, num_episodes=5)
            print('[iter {}] average score = {} (over 5 episodes)'.format(t, avg_score))
            evaluation_log = [t, avg_score]
            logger.writerow(evaluation_log)
        
        if t % snapshot_interval == 0:
            snapshot_path = save_path + 'iter{}_'.format(t)
            # save weight & training progress
            save_snapshot(snapshot_path, agent.critic, target_critic, critic_optim)
        
        if t > start_train and t % train_interval == 0:
            # start training after fixed number of steps
            # this may mitigate overfitting of networks to the 
            # small number of samples collected during the initial stage of training
            for _ in range(train_interval):
                update(agent,
                       replay_buf,
                       gamma,
                       critic_optim,
                       target_critic,
                       tau,
                       batch_size
                      )

    log_file.close()


###############################################
####### 3.Agent for greedy-algorithm  #########
###############################################
class greedyAgent:
    def __init__(self, obs_dim, num_act):
        self.obs_dim = obs_dim
        self.num_act = num_act
                
    def act(self, state, epsilon=0.0):
        return np.argmin(state)

################################
####### 4.Save result  #########
################################

#========================
# Save assignment result 
#========================
# For Q-learning agent
def save_assignment1(env,agent,info="None"):
    path = './'
    f = open(path+"assignment.txt",'a')

    s = env.reset()
    cumulative_reward = 0
    assignment = [[] for _ in range(env.num_gate)]
    reward_per_gate = [0]*env.num_gate

    for t in range(env.num_flight):
        a = agent.greedy(s)        
        s, r, _, _ = env.step(a)
        cumulative_reward += r
        assignment[a].append(t)
        reward_per_gate[a] += r

    f.write("\n\n=====  {}  =====".format(info))
    f.write("\nAssignment      : {}".format(assignment))
    f.write("\nReward_per_gate : {}".format(reward_per_gate))
    f.write("\nTotal_reward    : {}".format(cumulative_reward))
    f.close()

# For DQN, greedy agent
def save_assignment2(env,agent,info="None"):
    path = './'
    f = open(path+"assignment.txt",'a')

    s = env.reset()
    cumulative_reward = 0
    assignment = [[] for _ in range(env.num_gate)]
    reward_per_gate = [0]*env.num_gate

    for t in range(env.num_flight):
        a = agent.act(s)        
        s, r, _, _ = env.step(a)
        cumulative_reward += r
        assignment[a].append(t)
        reward_per_gate[a] += r

    f.write("\n\n=====  {}  =====".format(info))
    f.write("\nAssignment      : {}".format(assignment))
    f.write("\nReward_per_gate : {}".format(reward_per_gate))
    f.write("\nTotal_reward    : {}".format(cumulative_reward))
    f.close()

# For DQN, greedy agent
def write_datetime(strr: str):
    path = './'
    f = open(path+"assignment.txt",'a')
    f.write("\n\n\n**************  {}  Time: {}    **************".format(strr,datetime.now()))
    f.close()


#=====================
# Show & Save Results
#=====================
def print_result1(env,agent,silent=False):
    s = env.reset()
    cumulative_reward = 0
    for t in range(env.num_flight):
        if silent==False:
            print("\n==== stage {} ====".format(env.current_stage))
            print("arrival time :", env.arrival_time[env.current_stage])
            print("state :", env.to_ndim(s))
            print("W     :",env.w)
        
        a = agent.greedy(s)
        s, r, _, _ = env.step(a)
        cumulative_reward += r
        
        if silent==False:
            print("action:", a)
            print("reward:", r)

    print("\n=====  Final Report  =====")
    print("cumulative_reward: ", cumulative_reward)
    print("action distribution: ", np.unique(np.argmax(agent.Q,axis=1),return_counts=True))
    return(cumulative_reward)

def save_result1(env,agent,filename):
    path = '../log/'
    filename = filename
    f = open(path+filename+".txt",'w')

    s = env.reset()
    cumulative_reward = 0
    for t in range(env.num_flight):

        f.write("\n\n==== stage {} ====".format(env.current_stage))
        f.write("\narrival time : {}".format(env.arrival_time[env.current_stage]))
        f.write("\nstate : {}".format(env.to_ndim(s)))
        f.write("\nW     : {}".format(env.w))
        
        a = agent.greedy(s)
        s, r, _, _ = env.step(a)
        cumulative_reward += r
        
        f.write("\naction: {}".format(a))
        f.write("\nreward: {}".format(r))

    f.write("\n\n=====  Final Report  =====")
    f.write("\ncumulative_reward: {}".format(cumulative_reward))
    f.write("\naction distribution: {}".format(np.unique(np.argmax(agent.Q,axis=1),return_counts=True)))
    f.close()

def print_result2(env,agent,silent=False):
    s = env.reset()
    cumulative_reward = 0
    for t in range(env.num_flight):
        if silent==False:
            print("\n==== stage {} ====".format(env.current_stage))
            print("arrival time :", env.arrival_time[env.current_stage])
            print("state :", env.state)
        
        a = agent.act(s)
        s, r, _, _ = env.step(a)
        cumulative_reward += r
        
        if silent==False:
            print("action:", a)
            print("reward:", r)

    print("\n=====  Final Report  =====")
    print("cumulative_reward: ", cumulative_reward)
    return(cumulative_reward)

def save_result2(env,agent,filename):
    path = '../log/'
    filename = filename
    f = open(path+filename+".txt",'w')

    s = env.reset()
    cumulative_reward = 0
    for t in range(env.num_flight):

        f.write("\n\n==== stage {} ====".format(env.current_stage))
        f.write("\narrival time : {}".format(env.arrival_time[env.current_stage]))
        f.write("\nstate : {}".format(env.state))
        
        a = agent.act(s)
        s, r, _, _ = env.step(a)
        cumulative_reward += r
        
        f.write("\naction: {}".format(a))
        f.write("\nreward: {}".format(r))

    f.write("\n\n=====  Final Report  =====")
    f.write("\ncumulative_reward: {}".format(cumulative_reward))
    f.close()
