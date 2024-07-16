import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pandas as pd
from tools import whole_data_prepare_only_indicators, global_minmax_normalize
import os


print(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')

# Define the stock trading environment
class StockTradingEnv:
    def __init__(self, whole_data):
        self.whole_data = whole_data
        self.current_step = 55
        self.cash = 0
        self.curPos = np.zeros(50)
        self.totDVolume = 0
        self.value = 0
        self.todayPLL = []
        self.commRate = 0.001
        self.dlrPosLimit = 10000
        self.random_stock_num = np.random.randint(0, 50)
        self.a_stock_hist = whole_data[self.random_stock_num]
        self.look_back_length = 50
        self.my_moves_hist = deque([0] * self.look_back_length, maxlen=self.look_back_length)
        self.reset()

    def reset(self):
        self.current_step = 55
        self.cash = 0
        self.curPos = np.zeros(50)
        self.totDVolume = 0
        self.value = 0
        self.todayPLL = []
        self.random_stock_num = np.random.randint(0, 50)
        self.a_stock_hist = self.whole_data[self.random_stock_num]
        self.my_moves_hist = deque([0] * self.look_back_length, maxlen=self.look_back_length)
        return self._get_observation()

    def step(self, action):

        # 0: Sell, 1: Hold, 2: Buy
        def getPosition(action):
            if action == 0:  # Sell
                result_pos = np.zeros(50)
                result_pos[self.random_stock_num] = -100000
                return result_pos
            
            elif action == 1: # Hold
                return self.curPos
            
            elif action == 2:  # Buy
                result_pos = np.zeros(50)
                result_pos[self.random_stock_num] = 100000
                return result_pos
            
            elif action == 3: # Sell everything, back to 0
                result_pos = np.zeros(50)
                result_pos[self.random_stock_num] = 0
                return result_pos

        prcHistSoFar = self.whole_data[:, :self.look_back_length + self.look_back_length + self.current_step] # (50, 250) <class 'numpy.ndarray'>
        newPosOrig = getPosition(action)
        curPrices = prcHistSoFar[:, -1] # (50,) <class 'numpy.ndarray'>
        posLimits = np.array([int(x) for x in self.dlrPosLimit / curPrices]) # clip order limit
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - self.curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        self.totDVolume += dvolume # total money traded
        comm = dvolume * self.commRate # commision for trading on the day
        self.cash -= curPrices.dot(deltaPos) + comm
        self.curPos = np.array(newPos)
        posValue = self.curPos.dot(curPrices)
        todayPL = self.cash + posValue - self.value
        self.todayPLL.append(todayPL)
        self.value = self.cash + posValue
        ret = 0.0
        if (self.totDVolume > 0):
            ret = self.value / self.totDVolume
        
        self.current_step += 1
        done = self.current_step == len(self.a_stock_hist) - 1
        # reward = todayPL

        pll = np.array(self.todayPLL)
        (plmu, plstd) = (np.mean(pll), np.std(pll))

        reward = todayPL
        return self._get_observation(), reward, done

    def _get_observation(self):
        if (self.look_back_length > self.current_step):
            print('Error: Look back length > Current data length')
        inputs = whole_data_prepare_only_indicators(self.a_stock_hist[:self.look_back_length + self.look_back_length + self.current_step], seq_length=self.look_back_length)
        inputs = np.squeeze(inputs, axis=0)
        inputs = global_minmax_normalize(inputs, columns_to_normalize=[0,1,2,3,4,5,7,8,9,12])
        return np.append(inputs.flatten(), list(self.my_moves_hist))

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, action_size) 

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.save_dir = "RL_models3"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, episode):
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, f"{self.save_dir}/dqn_model_episode_{episode}.pth")
        print(f"Model saved at episode {episode}")

#=================================================================================================================================

def load_dqn_model(model_path, state_size, action_size, epsilon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = QNetwork(state_size, action_size).to(device)
    
    # Load the saved state
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epsilon = checkpoint['epsilon']
        episode = checkpoint['episode']
        
        print(f"Model loaded from episode {episode}")
        
        # Initialize the DQNAgent with the loaded model
        agent = DQNAgent(state_size, action_size)
        agent.model = model
        agent.epsilon = epsilon
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return agent, episode
    else:
        print(f"No saved model found at {model_path}")
        return None, None
    
state_size = 700  
action_size = 4  


commRate = 0.001
dlrPosLimit = 10000


def calcPL(prcHist, ep_num, start_day, end_day):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    todayPLL = []
    my_moves_hist = deque([0] * 50, maxlen=50)
    
    (_, nt) = prcHist.shape
    model_path = f"RL_models3/dqn_model_episode_{ep_num}.pth"
    loaded_agent, loaded_episode = load_dqn_model(model_path, state_size, action_size, epsilon=0)
    
    #for t in range(250, 501): # real
    for t in range(start_day, end_day):
        prcHistSoFar = prcHist[:, :t] # (50, 250) <class 'numpy.ndarray'>
        
        result_pos = np.zeros(nInst)
        for counter, stock in enumerate(prcHistSoFar):
            inputs = whole_data_prepare_only_indicators(stock, seq_length=50)
            inputs = np.squeeze(inputs, axis=0)
            inputs = global_minmax_normalize(inputs, columns_to_normalize=[0,1,2,3,4,5,7,8,9,12])
            inputs = np.append(inputs.flatten(), list(my_moves_hist))
            with torch.no_grad():
                action_values = loaded_agent.model(torch.Tensor(inputs).to(loaded_agent.device))
            predicted_action = torch.argmax(action_values).item()
            my_moves_hist.append(predicted_action)
            if predicted_action == 0:  # Sell
                result_pos[counter] = -100000

            elif predicted_action == 1: # Hold
                result_pos[counter] = curPos[counter]
            
            elif predicted_action == 2:  # Buy
                result_pos[counter] = 100000
            
            elif predicted_action == 3:  # Sell everything, back to 0
                result_pos[counter] = 0


        newPosOrig = result_pos
        curPrices = prcHistSoFar[:, -1] # (50,) <class 'numpy.ndarray'>
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # clip order limit
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume # total money traded
        comm = dvolume * commRate # commision for trading on the day
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        
        
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    
    return (plmu, ret, plstd, annSharpe, totDVolume)


#=================================================================================================================================






# Training loop
def train_dqn_agent(episodes, batch_size, save_interval):
    # Generate some dummy price data
    def loadPrices(fn):
        global nt, nInst
        df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
        (nt, nInst) = df.shape
        return (df.values).T


    pricesFile = "./prices1000.txt"
    price_data = loadPrices(pricesFile)
    price_data = price_data[:, :375]
    print(f'prcAll shape: {price_data.shape}')
    print("Loaded %d instruments for %d days" % (nInst, nt))

    

    env = StockTradingEnv(price_data)

    # nn = 2250
    # model_path_in = f"RL_models2/dqn_model_episode_{nn}.pth"
    # agent, loaded_episode = load_dqn_model(model_path_in, state_size, action_size, epsilon=0.05)
    agent = DQNAgent(state_size=700, action_size=4)
    total_reward_avg = 0

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            env.my_moves_hist.append(action)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_reward_avg += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                break


        if (e + 1) % save_interval == 0:

            print("=====")
            print(f'ep: {e+1}')

            agent.save_model(e + 1)

            print(f'Avg reward per episode: {total_reward_avg / save_interval}')
            total_reward_avg = 0

            def loadPrices_in(fn):
                df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
                return (df.values).T

            pricesFile_in = "./prices1000.txt"
            prcAll_in = loadPrices_in(pricesFile_in)
           
            (meanpl1, ret1, plstd1, sharpe1, dvol1) = calcPL(prcAll_in, ep_num=e+1, start_day=500, end_day=751)
            score1 = meanpl1 - 0.1*plstd1

            (meanpl2, ret2, plstd2, sharpe2, dvol2) = calcPL(prcAll_in, ep_num=e+1, start_day=750, end_day=1001)
            score2 = meanpl2 - 0.1*plstd2
            
            print("mean(PL): %.1f, %.1f" % (meanpl1, meanpl2))
            # print("return: %.5lf" % ret)
            print("StdDev(PL): %.2f, %.2f" % (plstd1, plstd2))
            # print("annSharpe(PL): %.2lf " % sharpe)
            print("totDvolume: %.0f, %.0f" % (dvol1, dvol2))
            print("Score: %.2f, %.2f" % (score1, score2))

            
            if score1 <= 0 and score2 <= 0:
                os.remove(f'{agent.save_dir}/dqn_model_episode_{e + 1}.pth')



if __name__ == "__main__": 
    # Run the training
    train_dqn_agent(episodes=5000, batch_size=512, save_interval=25)


'''
700 inputs

ep   | 0-375 | 375-501 |       500-751      |       750-1001 
===================================================================
750  |                 |  (57.1, 408) 16.20 |  (46.5, 445) 1.93
825  |                 |  (52.3, 433) 8.97  |  (50,   466) 3.36
850  |                 |  (57.6, 428) 14.69 |  (49.9, 466) 3.23
875  |                 |  (51,   435) 7.44  |  (46.4, 464) -0.05
900  |                 |  (58.3, 420) 16.19 |  (42.5, 454) -3.00
925  |                 |  (50,   421) 7.84  |  (38,   452) -7.29
950  |                 |  (53.6, 435) 9.99  |  (46.8, 467) 0.04
975  |                 |  (49.2, 443) 4.80  |  (40.2, 465) -6.26
1000 |                 |  (45.4, 440) 1.40  |  (47.9, 462) 1.65
1025 |                 |  (44,   435) 0.37  |  (48.7, 466) 2.07
1050 |                 |  (9.8,  419) -32.18|  (50,   401) 9.97
1075 |                 |  (45.6, 394) 6.16  |  (28.9, 406) -11.69
1100 |                 |  (43.7, 434) 3.80  |  (48.2, 463) 1.91
.... |                 |        ....        |       ....
1375 |                 | (0.5, 33.42) -2.86 |  (0, 0) 0              # totVolumn 30962 = 3 trades
1400 |                 |     (0, 0) 0       |  (0, 0) 0
'''