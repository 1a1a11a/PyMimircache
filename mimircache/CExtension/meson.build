project('mimircache', ['c', 'cpp'])
#project('mimircache')


######################### dependency #########################
glib_dep = dependency('glib-2.0', version: '>=2.40')
thread_dep = dependency('threads')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : true)


########################## sources ###########################
src_mimircache		= ['test.c']

src_cacheReader 	= ['cacheReader/binaryReader.c', 'cacheReader/csvReader.c', 'cacheReader/libcsv.c', 
						'cacheReader/reader.c', 'cacheReader/vscsiReader.c']

src_cache 			= ['cache/AMP.c', 'cache/LFUFast.c', 'cache/LRUSize.c', 'cache/PG.c', 'cache/Score.c', 'cache/ARC.c', 
						'cache/LRFU.c', 'cache/LRU_K.c', 'cache/Random.c', 'cache/cache.c', 'cache/FIFO.c', 'cache/LRU.c', 
						'cache/MRU.c', 'cache/SLRU.c', 'cache/mimir.c', 'cache/LFU.c', 'cache/LRUPage.c', 'cache/Optimal.c', 
						'cache/SLRUML.c']

src_profiler 		= ['profiler/LRUProfiler.c', 'profiler/eviction_stat.c', 'profiler/generalProfiler.c', 'profiler/heatmap.c', 
						'profiler/heatmap_related.c', 'profiler/heatmap_thread.c', 'profiler/partition.c', 'profiler/splay.c'] 

src_utils 			= ['utils/glib_related.c', 'utils/logging.c', 'utils/murmurhash3.c', 'utils/pqueue.c', 'utils/utils.c']

src_akamaiSimulator = ['akamaiSimulator/cacheLayer.cpp', 'akamaiSimulator/cacheLayerThread.cpp', 'akamaiSimulator/cacheServer.cpp',
						'akamaiSimulator/cacheServerThread.cpp', 'akamaiSimulator/consistentHashRing.cpp', 
						'akamaiSimulator/simulator.cpp', 'akamaiSimulator/simulatorMain.cpp'] 

src_libketama		= ['akamaiSimulator/libketama/ketama.c', 'akamaiSimulator/libketama/md5.c'] 


# for f in `ls */*.c`; do echo -n "'$f', "; done

########################## headers ##########################
inc_mimircache 				= 		include_directories('headers') 
inc_mimircache_cache 		= 		include_directories('cache/include')
inc_mimircache_cacheReader	= 		include_directories('cacheReader/include') 
inc_mimircache_profiler		= 		include_directories('profiler/include') 
inc_mimircache_utils		= 		include_directories('utils/include') 
inc_akamaiSimulator 		= 		include_directories('akamaiSimulator/include')
inc_libketama 				= 		include_directories('akamaiSimulator/libketama/include')


###################### target: library ####################### 
lib_cache = static_library('cache', sources:src_cache, dependencies: [glib_dep, thread_dep, m_dep], 
							include_directories: [inc_mimircache, inc_mimircache_cache, inc_mimircache_utils])

lib_cacheReader = static_library('cacheReader', sources:src_cacheReader, dependencies: [glib_dep, thread_dep, m_dep], 
							include_directories: [inc_mimircache, inc_mimircache_cache, inc_mimircache_cacheReader, inc_mimircache_utils])

lib_profiler = static_library('profiler', sources:src_profiler, dependencies: [glib_dep, thread_dep, m_dep], 
							include_directories: [inc_mimircache, inc_mimircache_cache, inc_mimircache_cacheReader,
												inc_mimircache_profiler, inc_mimircache_utils])

lib_utils = static_library('utils', sources:src_utils, dependencies: [glib_dep, thread_dep, m_dep], 
							include_directories: [inc_mimircache, inc_mimircache_cache, inc_mimircache_utils])

lib_ketama = static_library('libketama', sources:src_libketama, dependencies: [thread_dep, m_dep], 
							include_directories: inc_libketama) 


###################### target: executable ####################### 
executable('mimircache', sources:[src_mimircache], link_with: [lib_cache, lib_cacheReader, lib_profiler, lib_utils], 
	dependencies: [glib_dep, thread_dep, m_dep], include_directories: [inc_mimircache, inc_mimircache_cache, 
	inc_mimircache_cacheReader, inc_mimircache_profiler, inc_mimircache_utils])

executable('akamaiSimulator', sources:[src_akamaiSimulator], dependencies: [glib_dep, thread_dep, m_dep], 
	link_with: [lib_cache, lib_cacheReader, lib_profiler, lib_utils, lib_ketama], 
	include_directories: [inc_mimircache, inc_mimircache_cache, inc_mimircache_cacheReader, inc_mimircache_profiler, 
							inc_mimircache_utils, inc_akamaiSimulator, inc_libketama]) 



# rm * -r; meson --buildtype=debug ..; ninja; 