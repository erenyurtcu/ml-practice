import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000 # the simulation number can be a maximum of 10000
d = 10 # the number of different ads 

rewards = [0] *d # initially, all rewards are 0
total = 0 # total reward value
views = [0] * d # views of that time
selected = []
for n in range(0,N):
    ad = 0 # selected ad
    maxUCB = 0 # inital value of the max ucb is 0
    for i in range(0,d):
        if(views[i] > 0):
            mean = rewards[i] / views[i]
            delta = math.sqrt(3/2* math.log(n)/views[i])
            ucb = mean + delta
        else:
            ucb = N*10
        if maxUCB < ucb:
            maxUCB = ucb
            ad = i
            
        selected.append(ad)
        views[ad] = views[ad]+ 1
        reward = data.values[n,ad] # if the value in the n. cell is 1, the reward is 1
        rewards[ad] = rewards[ad]+ reward
        total = total + reward
        
print('Total Rewards: ' + str(total))   

# the plot
fig, ax = plt.subplots()
counts = [selected.count(i) for i in range(d)]
colors = plt.cm.tab10(np.linspace(0, 1, d))
bars = ax.bar(range(d), counts, color=colors, edgecolor='black', width=0.7)
ax.set_xlabel('Ads')
ax.set_ylabel('Number of Views')
ax.set_title('Number of Impressions by Ads')
ax.set_xticks(range(d))
plt.show()
