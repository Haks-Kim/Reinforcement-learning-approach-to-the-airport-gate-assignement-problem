import numpy as np
import random 

# 1.Environment for Q-Learning
def categorize(x,d_type):
    if d_type == 1:
        result = d1(x)
    elif d_type == 2:
        result = d2(x)
    elif d_type == 3:
        result = d3(x)
    elif d_type == 4:
        result = d4(x)
    else:
        result = d5(x)
    return result

# choice 1
def d1(x):
    if x == 0:
        g = 0 
    elif x > 0 and x <= 5 :
        g = 1
    else:
        g = 2   
    return g    

# choice 2
def d2(x):
    if x <= 5:
        g = 0 
    elif x > 5 and x <= 10 :
        g = 1
    else:
        g = 2   
    return g

# choice3
def d3(x):
    if x == 0:
        g = 0 
    elif x > 0 and x <= 10 :
        g = 1
    elif x >10 and x <=20 :
        g = 2
    else:
        g = 3   
    return g

# choice4
def d4(x):
    if x <= 10:
        g = 0 
    elif x > 10 and x <= 20 :
        g = 1
    elif x > 20 and x <= 40 :
        g = 2
    else:
        g = 3   
    return g

# choice5
def d5(x):
    if x <= 7:
        g = 0 
    elif x > 7 and x <= 14:
        g = 1
    elif x > 14 and x <= 21:
        g = 2
    elif x > 21 and x <= 28:
        g = 3  
    else:
        g = 4
    return g

class AGAP_Qlearning:
    def __init__(self,flight,gate,processing_time,arrival_time,d_type):
        self.num_flight = flight
        self.num_gate = gate
        
        self.processing_time = processing_time
        self.arrival_time = arrival_time
        self.w = np.zeros(self.num_gate).astype("int64") 
        
        self.d_type = int(d_type)
        self.num_category = int(3*(d_type==1) + 3*(d_type==2) + 4*(d_type==3) + 4*(d_type==4) + 5*(d_type==5) )
        self.current_stage = 0

    def update_w(self,action):
        t = self.current_stage
        for gate in range(self.num_gate):
            self.w[gate] = np.max([self.w[gate],self.arrival_time[t]]) + (action==gate)*self.processing_time[t]
        
    def get_reward(self,action):
        a = action
        t = self.current_stage
        return -max(0,self.w[a]-self.arrival_time[t])

    def update_state(self,action):     
        done = False
        if self.current_stage == self.num_flight-1:
            done = True
            next_state = [0] * self.num_gate
        else:
            next_state = []
            t = self.current_stage
            for gate in range(self.num_gate):
                if gate != action:
                    next_g = categorize(max(self.w[gate] - self.arrival_time[t+1],0),self.d_type )
                else:    
                    next_g = categorize(max(self.w[gate] + self.processing_time[t] \
                                        - (self.arrival_time[t+1] - self.arrival_time[t]),0), self.d_type)
                next_state.append(next_g)

        return next_state

    def step(self,action: int):
        done = False
        if self.current_stage == self.num_flight - 1:
            done = True

        reward = self.get_reward(action)

        self.update_w(action)
        _state = self.update_state(action)
        state = self.to_onedim(_state)

        self.current_stage += 1

        return state, reward, done, {}

    def reset(self):
        self.w = np.zeros(self.num_gate) 
        self.current_stage = 0
        return 0

    def to_onedim(self,list_):
        value = 0
        exponent= 0 
        for i in list_:
            value += i*self.num_category**exponent
            exponent += 1
        return value

    def to_ndim(self,value):
        list_ = []
        for i in reversed(range(self.num_gate)):
            quotient = value//self.num_category**i
            value = value - quotient*self.num_category**i
            list_ = [quotient] + list_
        return list_



# 2.Environment for DQN

class AGAP_DQN:
    def __init__(self,flight,gate,processing_time,arrival_time):
        self.num_flight = flight
        self.num_state = gate
        self.num_gate = gate
        
        self.processing_time = processing_time
        self.arrival_time = arrival_time

        self.state = np.zeros(self.num_gate)
        self.current_stage = 0

    def update_state(self,action):
        a = action
        t = self.current_stage
        for i in range(self.num_gate):
            self.state[i] = np.max( [self.state[i]+(a==i)*self.processing_time[t] - (self.arrival_time[t+1]-self.arrival_time[t]),0] )
        
    def get_reward(self,action):
        return -self.state[action]

    def step(self,action):
        done = False
        if self.current_stage == self.num_flight - 1:
            done = True

        reward = self.get_reward(action)

        if done == False:
            self.update_state(action)
            next_state = self.state
        else:
            next_state = np.zeros(self.num_gate)

        self.current_stage += 1

        return next_state, reward, done, {}

    def reset(self):
        self.state = np.zeros(self.num_gate) 
        self.current_stage = 0
        return self.state

