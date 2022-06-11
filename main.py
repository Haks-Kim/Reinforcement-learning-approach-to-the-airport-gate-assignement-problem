from agent import *
from environment import *

#==========================
# step1. Problem Instance #
#==========================
NF = 30
NG = 5 
np.random.seed (0)

Arriaval_Time1 = np.array([2*(i+1) for i in range(NF)]).astype("int64") 
Arriaval_Time2 = np.random.randint(0,5,30).cumsum().astype("int64")

Processing_Time1 = np.array([np.round(NF/NG) + 3 for _ in range(NF)]).astype("int64") 
Processing_Time2 = np.array([2*np.round(NF/NG) + 3 for _ in range(NF)]).astype("int64") 
Processing_Time3 = np.random.randint(10,21,30).astype("int64") 

# Instance1, Instance2 can be solved to optimality by greedy algorithm.
# Instance3 is difficult problem
Instance1 = (NF,NG,Processing_Time1,Arriaval_Time1)
Instance2 = (NF,NG,Processing_Time2,Arriaval_Time2)
Instance3 = (NF,NG,Processing_Time3,Arriaval_Time2)


#====================
# step2. Q-Learning #
#====================
'''
env1 = AGAP_Qlearning(*Instance1,d_type=1)
rollout_len = 100000
agent1 = Qlearning(env1, rollout_len = rollout_len)

print_result(env1,agent1,silent=False)
'''

#====================
# step3. Double-DQN #
#====================
gamma = 0.99
lr = 1e-6
tau = 1e-6
ep_len = 500
num_updates = 4000
batch_size = 128

env = AGAP_DQN(*Instance1)
agent = DQNAgent(obs_dim=NG,num_act=NG,hidden1=256,hidden2=256)
train(agent, env, gamma, 
      lr, tau,
      ep_len, num_updates, batch_size,
      init_buffer=5000, buffer_size=10000,
      start_train=2000, train_interval=50,
      eval_interval=2000, snapshot_interval=2000, path=None)

print_result2(env,agent,silent=False)
save_assignment2(env,agent)


agent = greedyAgent(obs_dim=NG, num_act=NG)
print_result2(env,agent,silent=False)
save_assignment2(env,agent,info="greedyAgent")
