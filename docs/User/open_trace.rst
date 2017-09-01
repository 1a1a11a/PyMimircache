.. _open_trace:

How to Open Different Traces
============================

Supported Trace File Type
-------------------------
- plain Text
- csv file
- binary file
- vscsi trace


How to Open a Trace File
------------------------
Now let's open a trace file, your have three choices for opening different types of trace files, choose the one suits your need.

>>> import mimircache as m
>>> c = m.cachecow()
>>> c.open("path/to/trace")
>>> c.csv("path/to/trace", init_params={'label':x})  # specify which column contains the request key(label)
>>> c.binary("path/to/trace", init_params={"label": x, "fmt": xxx})   # use same format as python struct
>>> c.vscsi("path/to/trace")                          # for vscsi format data


.. note::
    for csv and binary data, the column/field number begins from 1, so the first column(field) is 1, the second is 2, etc.
    In the init_params, other possible parameters are listed in the table below

    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | Keyword Argument | relavant file type   | Possible Value       | Default Value          | Description                                                    |
    +==================+======================+======================+========================+================================================================+
    | label            | csv/binary           | int                  | this is required       | the column of label of the request                             |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | fmt              | binary               | string               | this is required       | fmt string of binary data, same as python struct               |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | header           | csv                  | True/False           |      False             | whether csv data has header                                    |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | delimiter        | csv                  | char                 |        ","             | the delimiter separating fields in the csv file                |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | real_time        | csv/binary           | int                  |        NA              | the column of real time                                        |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | op               | csv/binary           | int                  |        NA              | the column of operation (read/write)                           |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+
    | size             | csv/binary           | int                  |        NA              | the column of block/request size                               |
    +------------------+----------------------+----------------------+------------------------+----------------------------------------------------------------+

OK, data is ready, now let's play!

If you want to read your data from cachecow, you can simply use cachecow as an iterator, for example, doing the following:

>>> for request in c:
>>>     print(c)

.. note::
    If you have a special data format, you can write your own reader in a few lines, see :ref:`here<advanced_usages>` about how to write your own cache reader.

