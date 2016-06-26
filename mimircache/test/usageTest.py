
from mimircache.top.cachecow import cachecow

m = cachecow(size=6000)
# m.test()
m.open('../data/parda.trace')
p = m.profiler("LRU")
# p = m.profiler(LRU, data='../data/parda.trace', dataType='plain')
p.heatmap()
# print(len(p.HRC))
# print(p.MRC[-1])
rdist = p.get_reuse_distance()
print(rdist)
# p.plotHRC()
# p.plotMRC()
