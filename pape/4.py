
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


dat_AMP = {"2000": 166063100/2006566, '6000': 167907200/1983684, '15000': 166589900/1930667, '32000':168821000/1897874}
dat_MS1 = {"2000": 234408900/3744374, '6000': 223239000/3127734, '15000': 211015300/2676918, '32000':201061800/2349418}
dat_MS2 = {"2000": 190752500/2408462, '6000': 184561100/2105470, '15000': 177605500/1939229, '32000':174316600/1857632}



# plt.bar(range(len(D)), D.values(), align='center')
# plt.xticks(range(len(D)), D.keys())
#
# plt.bar(range(len(dat_AMP)), dat_AMP.values(), align='center')




ind = 0.25+np.arange(3)  # the x locations for the groups
width = 0.2       # the width of the bars
print(ind)

fig = plt.figure()
ax = fig.add_subplot(111)

# yvals = np.array([dat_AMP["2000"], dat_MS1["2000"], dat_MS2["2000"]])
yvals = np.array([dat_AMP["6000"], dat_AMP["15000"], dat_AMP["32000"]])
rects1 = ax.bar(ind, yvals, width, color='r', align="center")
# y2vals = np.array([dat_AMP["6000"], dat_MS1["6000"], dat_MS2["6000"]])
y2vals = np.array([dat_MS1["6000"], dat_MS1["15000"], dat_MS1["32000"]])
rects2 = ax.bar(ind+width, y2vals, width, color='g', align="center")
# y3vals = np.array([dat_AMP["15000"], dat_MS1["15000"], dat_MS2["15000"]])
y3vals = np.array([dat_MS2["6000"], dat_MS2["15000"], dat_MS2["32000"]])
rects3 = ax.bar(ind+width*2, y3vals, width, color='b', align="center")
# y4vals = np.array([dat_AMP["32000"], dat_MS1["32000"], dat_MS2["32000"]])
# y4vals = np.array([dat_AMP["2000"], dat_AMP["6000"], dat_AMP["15000"], dat_AMP["32000"]])
# rects4 = ax.bar(ind+width*4, y3vals, width, color='y')

ax.set_xticks(ind+width)
ax.set_xticklabels( ('6000', '15000', '32000') )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('AMP', 'MS1', 'MS2'), loc="upper left")

plt.title("prefetching efficiency")
plt.xlabel("cache size/64K block")
plt.ylabel("prefetching efficiency%")
# plt.gca().set_yscale('log')
plt.ylim(ymin=60)

plt.savefig("test.png", dpi=600)