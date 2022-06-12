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
Processing_Time3 = np.random.randint(4,15,30).astype("int64") 
Processing_Time4 = np.random.randint(10,21,30).astype("int64") 

# Instance1, Instance2 can be solved to optimality by greedy algorithm.
# Instance3 is difficult problem
Instance1 = (NF,NG,Processing_Time1,Arriaval_Time1)
Instance2 = (NF,NG,Processing_Time2,Arriaval_Time1)
Instance3 = (NF,NG,Processing_Time3,Arriaval_Time2)
Instance4 = (NF,NG,Processing_Time4,Arriaval_Time2)
Instance_list = [Instance1,Instance2,Instance3,Instance4]
instance_num = 4

write_datetime("Start")




#====================
# step2. Q-Learning #
#====================
# Info naming rule: instance_method_dtype for qlearning
info_list = ["Instance1_qlearning_d1","Instance1_qlearning_d2","Instance1_qlearning_d3","Instance1_qlearning_d4","Instance1_qlearning_d5"]\
         + ["Instance2_qlearning_d1","Instance2_qlearning_d2","Instance2_qlearning_d3","Instance2_qlearning_d4","Instance2_qlearning_d5"]\
         + ["Instance3_qlearning_d1","Instance3_qlearning_d2","Instance3_qlearning_d3","Instance3_qlearning_d4","Instance3_qlearning_d5"]\
         + ["Instance4_qlearning_d1","Instance4_qlearning_d2","Instance4_qlearning_d3","Instance4_qlearning_d4","Instance4_qlearning_d5"]

env_list = [AGAP_Qlearning(*Instance1,1),AGAP_Qlearning(*Instance1,2),AGAP_Qlearning(*Instance1,3),AGAP_Qlearning(*Instance1,4),AGAP_Qlearning(*Instance1,5)] \
        + [AGAP_Qlearning(*Instance2,1),AGAP_Qlearning(*Instance2,2),AGAP_Qlearning(*Instance2,3),AGAP_Qlearning(*Instance2,4),AGAP_Qlearning(*Instance2,5)] \
        + [AGAP_Qlearning(*Instance3,1),AGAP_Qlearning(*Instance3,2),AGAP_Qlearning(*Instance3,3),AGAP_Qlearning(*Instance3,4),AGAP_Qlearning(*Instance3,5)]\
        + [AGAP_Qlearning(*Instance4,1),AGAP_Qlearning(*Instance4,2),AGAP_Qlearning(*Instance4,3),AGAP_Qlearning(*Instance4,4),AGAP_Qlearning(*Instance4,5)]

rollout_len1,rollout_len2,rollout_len3 = (2000000,8000000,40000000)       
rollout_list = [rollout_len1,rollout_len1,rollout_len2,rollout_len2,rollout_len3]*4

agent_list = []

for i in [4,9,14,19]:
    print("\nExperiment Info: ", info_list[i])
    agent = Qlearning(env_list[i], rollout_len = rollout_list[i])
    #agent_list.append(Qlearning(env_list[i], rollout_len = rollout_list[i]))
    save_assignment1(env_list[i],agent,info_list[i])
    reward = print_result1(env_list[i],agent,silent=True)
    print(" ")

'''
#====================
# step3. Double-DQN #
#====================
gamma = 0.99
lr = 1e-6
tau = 1e-6
ep_len = 500
num_updates = 40000
batch_size = 128

info_list = ["Instance1_DQN","Instance2_DQN","Instance3_DQN","Instance4_DQN"]

for i in range(instance_num):
    print("\nExperiment Info: ", info_list[i])
    agent = DQNAgent(obs_dim=NG,num_act=NG,hidden1=256,hidden2=256)
    env = AGAP_DQN(*Instance_list[i])
    train(agent, env, gamma, 
          lr, tau,
          ep_len, num_updates, batch_size,
          init_buffer=5000, buffer_size=10000,
          start_train=2000, train_interval=50,
          eval_interval=2000, snapshot_interval=2000, path=None)

    save_assignment2(env,agent,info_list[i])
    reward = print_result2(env,agent,silent=True)
    print(" ")

#====================
# step4 Greedy-agent 
#====================
info_list = ["Instance1_greedy","Instance2_greedy","Instance3_greedy","Instance4_greedy"]
agent = greedyAgent(obs_dim=NG, num_act=NG)

for i in range(instance_num): 
    print("\nExperiment Info: ", info_list[i])
    env = AGAP_DQN(*Instance_list[i])
    save_assignment2(env,agent,info_list[i])
    reward = print_result2(env,agent,silent=True)
    print(" ")

write_datetime("Completion")

'''