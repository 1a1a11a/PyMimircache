from mimircache.top.mimircache import cacheCow

m = cacheCow(size=10000)
# m.test()
m.open('../data/parda.trace')
p = m.profiler("LRU")
# p = m.profiler(LRU, data='../data/parda.trace', dataType='plain')
# p.run()
# print(len(p.HRC))
# print(p.MRC[-1])
rdist = p.get_reuse_distance()
print(rdist[-1])
# p.plotHRC()
# p.plotMRC()
