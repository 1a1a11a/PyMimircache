.. _quick_start:

Quick Start
===========

Get Prepared
------------
With mimircache, testing/profiling cache replacement algorithms is very easy.
Let's begin by getting a cachecow object from mimircache:

>>> import mimircache as m
>>> c = m.cachecow()

Open Trace File
---------------
Now let's open a trace file, your have three choices for opening different types of trace files, choose the one suits your need.


>>> c.open("trace/file/location")
>>> c.csv("trace/file/location", init_params={'label':x})  # specify which column contains the request key(label)
>>> c.vscsi("trace/file/location")          # for vscsi format data
>>> c.binary("trace/file/location", init_params={"label": x, "fmt": xxx})   # use same format as python struct



see :ref:`here<open_trace>` for details.


Get Basic Statistics
--------------------
You can get some statistics about the trace, for example how many request, how may unique requests.

+-------------------+---------------+------------------------------------------------------------+
| Functions         | Parameters    |       Description                                          |
+-------------------+---------------+------------------------------------------------------------+
| num_of_req        | None          | return the number of requests in the trace                 |
+-------------------+---------------+------------------------------------------------------------+
| num_of_uniq_req   | None          | return the number of unique requests in the trace          |
+-------------------+---------------+------------------------------------------------------------+
| reset             | None          | reset reader to the beginning of the trace                 |
+-------------------+---------------+------------------------------------------------------------+
| __len__           | None          | return the number of requests in the trace                 |
+-------------------+---------------+------------------------------------------------------------+

If you want to read your data from cachecow, you can simply use cachecow as an iterator, for example, doing the following:

>>> for request in c:
>>>     print(c)



Profiler and Profiling
______________________

With a profiler, you can obtain the reuse distance of a request, the hit count and hit ratio of at a certain size, you can even directly plot the hit ratio curve(HRC).
See :ref:`here<profiling>` for details.



Basic Plotting
______________

Basic plotting describes how to use mimircache to plot cold_miss, cold_miss_ratio, request_num, obj_mapping plot and hit ratio curve (miss ratio curve) plots.
See :ref:`here<basic_plotting>` for details.




Heatmap Plotting
________________

Heatmap plotting section describes how to use mimircache to plot heatmaps.
See :ref:`here<heatmap_plotting>` for details.







Congratulations! You have finished the basic tutorial! Check :ref:`Advanced Usage<advanced_usages>` part if you need.