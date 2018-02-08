.. _installation:

Installation
============

This part of the documentation covers the installation of PyMimircache. PyMimircache currently has the following dependencies:

**pkg-config , glib , scipy , numpy , matplotlib**,

PyMimircache has been tested on Python3.4, Python3.5, Python3.6.
**Mac Users**: if you don't know how to install these packages, try macports or homebrew; google will help you.


General Installation (pip)
--------------------------
Using pip3 is the preferred way for installing PyMimiracache, notice that PyMimircache does not support python2.

First Step: Install C Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First: use any package management software to install pkg-config and glib.


Second Step: Install Python Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use pip3 to install python dependencies::

$ sudo pip3 install matplotlib heapdict


Third Step: pip Install PyMimircache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install PyMimircache, simply run this command in your terminal of choice::

$ sudo pip3 install PyMimircache




Install From Source
-------------------
This is an alternative method, only use this one when you can't install using the methods above, or you want to try the newest feature of PyMimircache.
Beware that it might have bugs in the newest version. We highly recommend that you to use the stable version from pip3.

Install All Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^
Installing all dependencies is the same as in General Installation .

Get the Source Code
^^^^^^^^^^^^^^^^^^^^
PyMimircache is actively developed on GitHub, where the code is
always available `here <https://github.com/1a1a11a/PyMimircache/tree/master>`_.

You can clone the public repository::

    $ git clone -b master --recurse-submodules git://github.com/1a1a11a/PyMimircache.git

Once you have a copy of the source, you can install it into your site-packages easily::

    $ sudo python3 setup.py install




Install Using Docker Container
------------------------------
As an alternative, you can use PyMimircache in a docker container,

Use interactive shell
^^^^^^^^^^^^^^^^^^^^^

To enter an interactive shell and do plotting, you can run::

    sudo docker run -it --rm -v $(pwd):/PyMimircache/scripts -v PATH/TO/DATA:/PyMimircache/data 1a1a11a/PyMimircache /bin/bash

After you run this command, you will be in a shell with everything ready, your current directory is mapped to `/PyMimircache/scripts/` and your data directory is mapped to `/PyMimircache/data`. In addition, we have prepared a test dataset for you at `/PyMimircache/testData`.

 

Run scripts directly
^^^^^^^^^^^^^^^^^^^^

If you don't want to use an interactive shell and you have your script ready, then you can do::

    docker run --rm -v $(pwd):/PyMimircache/scripts -v PATH/TO/DATA:/PyMimircache/data 1a1a11a/PyMimircache python3 /PyMimircache/scripts/YOUR_PYTHON_SCRIPT.py

However, if you are new here or you have trouble using docker to run scripts directly, we suggest using an interactive shell which can help you debug.





Special Instructions for Installing on Ubuntu
---------------------------------------------

First Step: Install Python3, pip and all dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

** If you are using Ubuntu 12**, you need to do the first step.
Since Ubuntu 12 does not come with Python3 or above and some related components like pip,
you can either compile Python3 from source or add the repository from Ubuntu 14 into your source list.
Below are the instructions for using Ubuntu 14 repository to install Python3 and pip.

Add the following two lines to the top of /etc/apt/source.list
::

    deb http://us.archive.ubuntu.com/ubuntu/ trusty main restricted universe multiverse
    deb-src http://us.archive.ubuntu.com/ubuntu/ trusty main restricted universe multiverse

Then update your repository by::

$ sudo apt-get update

**If you are using Ubuntu 14 and above, start here, Ubuntu 12 continues here**:

.. _Ubuntu 14 start here:

Now install python3, python3-pip and all dependencies::

$ sudo apt-get install python3 python3-pip python3-matplotlib pkg-config libglib2.0-dev

Second Step: Install PyMimircache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

$ sudo pip3 install PyMimircache

**Congratulations! You have finished installation, welcome to the world of PyMimircache**