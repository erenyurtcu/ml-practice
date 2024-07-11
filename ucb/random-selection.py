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
plt.hist(selected)
plt.show()