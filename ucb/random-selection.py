import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data = pd.read_csv('Ads_CTR_Optimisation.csv')



N = 10000 # the simulation number can be a maximum of 10000
d = 10 # the number of different ads 
total = 0 # the initial value of total should be 0
selected = []
for n in range(0,N):
    ad = random.randrange(d)
    selected.append(ad)
    reward  = data.values[n,ad] # if the value in the n. cell is 1, the reward is 1
    total = total + reward # if it has a reward, increase the total by 1
    
# the plot of ads
fig, ax = plt.subplots()
counts = [selected.count(i) for i in range(d)]
colors = plt.cm.tab10(np.linspace(0, 1, d))
bars = ax.bar(range(d), counts, color=colors, edgecolor='black', width=0.7)
ax.set_xlabel('Ads')
ax.set_ylabel('Number of Views')
ax.set_title('Number of Impressions by Ads')
ax.set_xticks(range(d))
plt.show()