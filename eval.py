#!/usr/bin/env python

import numpy as np
import pandas as pd
from Quartz import getMyPosition as getPosition
import matplotlib.pyplot as plt


nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


pricesFile = "./prices750.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))


start_day = 500
end_day = 751

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
        newPosOrig = getPosition(prcHistSoFar)
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
        
      
        # thing I added for info

        # if t % 100 == 0:
        #     print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
        #         (t, value, todayPL, totDVolume, ret))
        #     print(f'curPos: {curPos} {type(curPos)}\n\n')
        # print('\n\n')
        # print(t)
        # if t == 378:
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


