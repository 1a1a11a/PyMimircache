.. _quick_start:

Quick Start
===========

Get Prepared
------------
With PyMimircache, testing/profiling cache replacement algorithms is very easy.
Let's begin by getting a cachecow object from PyMimircache:

>>> from PyMimircache import Cachecow
>>> c = Cachecow()

Open Trace File
---------------
Now let's open a trace file. You have three choices for opening different types of trace files. Choose the one that suits your needs.


>>> c.open("trace/file/location")
>>> c.csv("trace/file/location", init_params={'label':x})  # specify which column contains the request key(label)
>>> c.vscsi("trace/file/location")          # for vscsi format data
>>> c.binary("trace/file/location", init_params={"label": x, "fmt": xxx})   # use same format as python struct



see :ref:`here<open_trace>` for details.


Get Basic Statistics
--------------------
You can get some statistics about the trace, for example how many request, how may unique requests.

+-------------------------+--------------------------------+------------------------------------------------------------+
| Functions               | Parameters                     |       Description                                          |
+-------------------------+--------------------------------+------------------------------------------------------------+
| num_of_req              | None                           | return the number of requests in the trace                 |
+-------------------------+--------------------------------+------------------------------------------------------------+
| num_of_uniq_req         | None                           | return the number of unique requests in the trace          |
+-------------------------+--------------------------------+------------------------------------------------------------+
| stat                    | None                           | get a list of statistical information about the trace      |
+-------------------------+--------------------------------+------------------------------------------------------------+
| characterize            | type (short/medium/long/all)   | plot a series of fig, type indicates run time              |
+-------------------------+--------------------------------+------------------------------------------------------------+
| len                     | None                           | return the number of requests in the trace                 |
+-------------------------+--------------------------------+------------------------------------------------------------+

If you want to read your data from cachecow, you can simply use cachecow as an iterator, for example, doing the following:

>>> for request in c:
>>>     print(c)



Profiler and Profiling
-----------------------
Now cachecow supports basic profiling, to conduct complex profiling, you still need to get a profiler.
With a profiler, you can obtain the reuse distance of a request, the hit count and hit ratio of at a certain size, you can even directly plot the hit ratio curve (HRC). See :ref:`here<profiling>` for details.

cachecow supports two type of profiling right now, calculate reuse distance and calculate hit ratio. The syntax is listed below.

>>> # get an array of reuse distance
>>> c.get_reuse_distance()
>>> # get a dictionary of cache size -> hit ratio
>>> c.get_hit_ratio_dict(algorithm, cache_size=-1, cache_params=None, bin_size=-1)

See :ref:`API-cachecow<API>` section for details.


Two Dimensional Plotting
------------------------

cachecow supports the following two dimensional figures,
        ========================  ============================  =================================================
                plot type               required parameters         Description
        ========================  ============================  =================================================
            cold_miss_count         time_mode, time_interval     cold miss count VS time
            cold_miss_ratio         time_mode, time_interval     coid miss ratio VS time
            request_rate            time_mode, time_interval     num of requests VS time
            popularity              NA                           Percentage of obj VS frequency
            rd_popularity           NA                           Num of req VS reuse distance
            rt_popularity           NA                           Num of req VS reuse time
            mapping                 NA                           mapping from original objID to sequential number
          interval_hit_ratio        cache_size                   hit ratio of interval VS time
        ========================  ============================  =================================================

The basic syntax for plotting the two dimensional figures is here

>>> # see table for plot_type names
>>> c.twoDPlot(plot_type, **kwargs)


See :ref:`API-twoDPlots<API>` section and :ref:`basic plotting<basic_plotting>` for details.


Hit Ratio Curve Plotting
------------------------

cachecow supports plotting against a list of cache replacement algorithms, using the following syntax:

>>> plotHRCs(algorithm_list, cache_params=(), cache_size=-1, bin_size=-1, auto_resize=True, figname="HRC.png", **kwargs)

See :ref:`API-LRUProfiler and API-cGeneralProfiler<API>` section and :ref:`basic plotting<basic_plotting>` for details.


Heatmap Plotting
----------------

cachecow supports basic heatmap plotting, and supported plot type is listed below.

>>> # plot heatmaps
>>> heatmap(time_mode, plot_type, time_interval=-1, num_of_pixels=-1, algorithm="LRU", cache_params=None, cache_size=-1, **kwargs)
>>> # plot differential heatmaps
>>> diff_heatmap(time_mode, plot_type, algorithm1, time_interval=-1, num_of_pixels=-1, algorithm2="Optimal", cache_params1=None, cache_params2=None, cache_size=-1, **kwargs)

================================================================  ========================================================================
    plot type                                                           Description
================================================================  ========================================================================
    * hit_ratio_start_time_end_time                                     Hit ratio heatmap of given start time and end time
    * hit_ratio_start_time_cache_size (python only)                     Hit ratio heatmap of given start time and cache size
    * avg_rd_start_time_end_time (python only)                          Average reuse distance of start time and end time
    * cold_miss_count_start_time_end_time (python only)                 deprecated
    * rd_distribution                                                   Heatmap of reuse distance distribution over time
    * rd_distribution_CDF                                               Heatmap (CDF) of reuse distance distribution over time
    * future_rd_distribution                                            Heatmap of future reuse distribution over time
    * dist_distribution                                                 Heatmap of distance distribution over time
    * reuse_time_distribution                                           Heatmap of reuse time distribution over time
================================================================  ========================================================================

Heatmap plotting section describes how to use PyMimircache to plot heatmaps.
See :ref:`API-cHeatmap<API>` section and :ref:`here<heatmap_plotting>` for details.




Congratulations! You have finished the basic tutorial! Check :ref:`Advanced Usage<advanced_usages>` part if you need.