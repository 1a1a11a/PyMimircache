import csv
import numpy as np
import matplotlib.pyplot as plt

hr_dic = {}
with open('dat/large.csv', 'r') as ifile:
    reader = csv.reader(ifile)
    header = next(reader)
    for name in header:
        hr_dic[name] = []
    for line in reader:
        hr_dic['cache-size'].append(float(line[0]))
        hr_dic['ARC'].append(float(line[1]))
        hr_dic['clock'].append(float(line[2]))
        hr_dic['FIFO'].append(float(line[3]))
        # hr_dic['LFU_MRU'].append(float(line[4]))
        # hr_dic['LFU_LRU'].append(float(line[5]))
        # hr_dic['LFU_RR'].append(float(line[6]))
        # hr_dic['LRU'].append(float(line[7]))
        hr_dic['MRU'].append(float(line[8]))
        hr_dic['RR'].append(float(line[9]))
        hr_dic['SLRU'].append(float(line[10]))
        hr_dic['S4LRU'].append(float(line[11]))

plt.figure(1, figsize=(6, 8), dpi=300)

plt.subplot(4, 2, 1)
# print(hr_dic['ARC'])
print(len(hr_dic['cache-size']))
print(len(hr_dic['ARC']))
plt.plot(np.array(hr_dic['cache-size']), np.array(hr_dic['ARC']), label='ARC', color='red')
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

plt.subplot(4, 2, 2)
plt.plot(hr_dic['cache-size'], hr_dic['clock'], label='clock', color='green')
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

plt.subplot(4, 2, 3)
plt.plot(hr_dic['cache-size'], hr_dic['FIFO'], label='FIFO', color='cyan')
plt.ylim(0, 60)
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

# plt.subplot(4,3,4)
# plt.plot(hr_dic['cache-size'], hr_dic['LFU_MRU'], label='LFU_MRU', color='magenta')
# plt.ylim(0, 60)
# plt.yticks([])
# plt.xticks([])
# plt.legend(fancybox=True, frameon=False, loc="upper right", fontsize=12)
#
# plt.subplot(4,3,5)
# plt.plot(hr_dic['cache-size'], hr_dic['LFU_LRU'], label='LFU_LRU', color='yellow')
# plt.ylim(0, 60)
# plt.yticks([])
# plt.xticks([])
# plt.legend(fancybox=True, frameon=False, loc="upper right", fontsize=12)
#
# plt.subplot(4,3,6)
# plt.plot(hr_dic['cache-size'], hr_dic['LFU_RR'], label='LFU_RR', color='black')
# plt.ylim(0, 60)
# plt.yticks([])
# plt.xticks([])
# plt.legend(fancybox=True, frameon=False, loc="upper right", fontsize=12)
#
# plt.subplot(4,3,7)
# plt.plot(hr_dic['cache-size'], hr_dic['LRU'], label='LRU', color='red')
# plt.ylim(0, 60)
# plt.yticks([])
# plt.xticks([])
# plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)
#
plt.subplot(4, 2, 4)
plt.plot(hr_dic['cache-size'], hr_dic['MRU'], label='MRU', color='green')
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

plt.subplot(4, 2, 5)
plt.plot(hr_dic['cache-size'], hr_dic['RR'], label='Random', color='blue')
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

plt.subplot(4, 2, 6)
plt.plot(hr_dic['cache-size'], hr_dic['SLRU'], label='SLRU', color='cyan')
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

plt.subplot(4, 2, 7)
plt.plot(hr_dic['cache-size'], hr_dic['S4LRU'], label='S4LRU', color='magenta')
plt.ylim(0, 60)
plt.yticks([])
plt.xticks([])
plt.legend(fancybox=True, frameon=False, loc="lower right", fontsize=12)

plt.subplot(4, 2, 8)
plt.text(0.5, 0.5, 'in all figures, \ncache-size is 800 \nitems, and all hit \nrate is normalized\n',
         horizontalalignment='center',
         verticalalignment='center', withdash=True, fontdict={})
plt.gca().patch.set_facecolor('yellow')
plt.yticks([])
plt.xticks([])

plt.gcf().suptitle("Hit rate on large data", fontsize=20)
plt.gcf().subplots_adjust(top=0.92)

# plt.subplot(4,3,12)
# plt.plot(hr_dic['cache-size'], hr_dic['ARC'])

# plt.show()
plt.savefig("large1.png")

plt.figure(2, dpi=300)
plt.plot(hr_dic['cache-size'], hr_dic['ARC'], 'r', label='ARC', color='red', linewidth=2)

plt.plot(hr_dic['cache-size'], hr_dic['clock'], 'r', label='clock', color='green', linewidth=2)
plt.plot(hr_dic['cache-size'], hr_dic['FIFO'], 'r', label='FIFO', color='cyan', linewidth=2)
# plt.plot(hr_dic['cache-size'], hr_dic['LFU_LRU'], 'r', label='LFU_LRU')
# plt.plot(hr_dic['cache-size'], hr_dic['LFU_MRU'], 'r', label='LFU_MRU')
# plt.plot(hr_dic['cache-size'], hr_dic['LFU_RR'], 'r', label='LFU_RR')
# plt.plot(hr_dic['cache-size'], hr_dic['LRU'], 'r', label='LRU', color='magenta', linewidth=2)
plt.plot(hr_dic['cache-size'], hr_dic['MRU'], 'r', label='MRU', color='magenta', linewidth=2)
plt.plot(hr_dic['cache-size'], hr_dic['RR'], 'r', label='RR', color='yellow', linewidth=2)
plt.plot(hr_dic['cache-size'], hr_dic['SLRU'], 'r', label='SLRU', color='black', linewidth=2)
plt.plot(hr_dic['cache-size'], hr_dic['S4LRU'], 'r', label='S4LRU', color='orange', linewidth=2)
plt.xlim(0, 2000000)
plt.ylim(0, 60)
plt.title("hit rate curve", fontsize=18)
plt.xlabel("cache size/item", fontsize=14)
plt.ylabel("hit rate/%", fontsize=14)
plt.legend(loc="best", numpoints=8)

# plt.show()
plt.savefig('large_2.png')
