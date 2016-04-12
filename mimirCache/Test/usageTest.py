from mimirCache.Top.cacheCow import cacheCow

m = cacheCow(size=10000)
# m.test()
m.open('../Data/parda.trace')
p = m.profiler("LRU")
# p = m.profiler(LRU, data='../Data/parda.trace', dataType='plain')
# p.run()
# print(len(p.HRC))
# print(p.MRC[-1])
rdist = p.get_reuse_distance()
print(rdist[-1])
# p.plotHRC()
# p.plotMRC()
