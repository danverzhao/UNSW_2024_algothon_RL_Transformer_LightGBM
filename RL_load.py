import torch
from collections import deque
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RL import QNetwork, DQNAgent
from tools import whole_data_prepare_only_indicators, global_minmax_normalize



def load_dqn_model(model_path, state_size, action_size):
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
        agent.epsilon = 0 # 
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return agent, episode
    else:
        print(f"No saved model found at {model_path}")
        return None, None


state_size = 650  
action_size = 3  
'''
model_path = "RL_models/dqn_model_episode_5.pth"

loaded_agent, loaded_episode = load_dqn_model(model_path, state_size, action_size)

if loaded_agent:
    print(f"Agent loaded successfully from episode {loaded_episode}")
    # You can now use loaded_agent for inference or continue training
    
    # Example: Using the loaded model for inference
    sample_state = torch.rand(state_size)  # Create a sample state
    with torch.no_grad():
        action_values = loaded_agent.model(sample_state.unsqueeze(0).to(loaded_agent.device))
    predicted_action = torch.argmax(action_values).item()
    print(f"Predicted action for sample state: {predicted_action}")
else:
    print("Failed to load the agent")
'''




def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


pricesFile = "./prices1000.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))

ep_num = 2350
model_path = f"RL_models/dqn_model_episode_{ep_num}.pth"
loaded_agent, loaded_episode = load_dqn_model(model_path, state_size, action_size)

start_day = 750
end_day = 1001

'''
     |   105 - 375   |   375 - 501   |   500 - 751   |   375 - 751
-------------------------------------------------------------------
750  | 84.6 418 42.75| 65.2 403 24.89| 62.7 427 19.92| 66.7 426 24
960  | 109 446 64.81 |19.3 427 -23.5 | 53.3 440 9.32 | 39.8 438 -4
1050 | 35 414 -6.45  | 30.6 364 -5.93| 52.5 373 15.2 | 43.8 373 6.45
2250 | 93.5 400 53.41| 44.5 302 14.2 | 49.4 366 12.75| 43.8 369 6.93

'''

def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    
    
    #for t in range(250, 501): # real
    for t in range(start_day, end_day):
        prcHistSoFar = prcHist[:, :t] # (50, 250) <class 'numpy.ndarray'>
        
        result_pos = np.zeros(nInst)
        for counter, stock in enumerate(prcHistSoFar):
            inputs = whole_data_prepare_only_indicators(stock, seq_length=50)
            inputs = np.squeeze(inputs, axis=0)
            inputs = global_minmax_normalize(inputs, columns_to_normalize=[0,1,2,3,4,5,7,8,9,12])
            inputs = inputs.flatten()
            with torch.no_grad():
                action_values = loaded_agent.model(torch.Tensor(inputs).to(loaded_agent.device))
            predicted_action = torch.argmax(action_values).item()
            if predicted_action == 0:  # Sell
                result_pos[counter] = -100000

            elif predicted_action == 1: # Hold
                result_pos[counter] = curPos[counter]
            
            elif predicted_action == 2:  # Buy
                result_pos[counter] = 100000

       
        newPosOrig = result_pos
        curPrices = prcHistSoFar[:, -1] # (50,) <class 'numpy.ndarray'>
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # clip order limit
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        #print(f'day {t}: {newPos}')
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
        
      
        # thing I added for info

        # if t % 100 == 0:
        #     print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
        #         (t, value, todayPL, totDVolume, ret))
        #     print(f'curPos: {curPos} {type(curPos)}\n\n')
        # print('\n\n')
        # print(t)
        # if t == 515:
        #     break
        
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    
    plt.figure(figsize=(12, 6))
    plt.plot(todayPLL)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    cumulative_sum = [sum(todayPLL[:i+1]) for i in range(len(todayPLL))]
    plt.plot(cumulative_sum)
    plt.show()
    return (plmu, ret, plstd, annSharpe, totDVolume)

if __name__ == "__main__":
    total_score = 0
    how_many_differnt_starting_days = 1
    for length in range(how_many_differnt_starting_days):
        (meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
        score = meanpl - 0.1*plstd
        print("=====")
        print(f'start day: {start_day}, end day: {end_day}')
        print("mean(PL): %.1lf" % meanpl)
        # print("return: %.5lf" % ret)
        print("StdDev(PL): %.2lf" % plstd)
        # print("annSharpe(PL): %.2lf " % sharpe)
        print("totDvolume: %.0lf " % dvol)
        print("Score: %.2lf" % score)
        total_score += score
        start_day += 1

    print(f'average score: {round(total_score/how_many_differnt_starting_days, 2)}')

