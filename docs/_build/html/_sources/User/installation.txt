.. _installation:

Installation
============
.. warning:: 
	mimircache has been tested on python3.4 and python3.5. Theoretically, all python3 are supported, but if you are using python 3.1~3.2, you might have a hard time install pip, here we suggest using the latest version of python3.

This part of the documentation covers the installation of mimircache. Mimircache currently has the following dependencies:
**pkg-config, glib, scipy, numpy, matplotlib**

General Installation(Pip)
--------------------------
First Step: Install C Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The first step is to use any package management software to install pkg-config and glib.

.. note:: **Mac Users**: if you don't know how to install these packages, try macports or homebrew, google will help you.

Second Step: Install Python Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using pip3 to install python dependencies::

$ sudo pip3 install matplotlib


Third Step: Pip Install mimircache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install mimircache, simply run this command in your terminal of choice::

$ sudo pip3 install mimircache


Install From Source
-------------------
This is an alternative method, only use this one when you can't install using the methods above, or you want to try the newest feature of mimircache, notice that, it might have bugs in newest version, we highly recommend you to use the stable version from pip3.

Install All Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^
Installing all dependencies is the same as the one described above.

Get the Source Code
^^^^^^^^^^^^^^^^^^^^
mimircache is actively developed on GitHub, where the code is
always available `here <https://github.com/1a1a11a/mimircache/tree/master>`_.

You can clone the public repository::

    $ git clone -b master git://github.com/1a1a11a/mimircache.git

Once you have a copy of the source, you can install it into your site-packages easily::

    $ sudo python3 setup.py install


Special Instructions for Installing on Ubuntu
------------------------------------------------

First Step: Install Python3, pip and all dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
    If you are using Ubuntu 14 and above, you can `jump to here <Ubuntu 14 start here_>`_

However, if you are using Ubuntu 12, you need to do the first step.
Since Ubuntu 12 does not come with Python3 or above and some related components like pip. You can either compile python3 from source or add the repository from Ubuntu 14 into your source list. Below is the instructions for using Ubuntu 14 repository to install python3 and pip.

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

Second Step: Install mimircache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

$ sudo pip3 install mimircache

**Congratulations! You have finished installation, welcome to the world of mimircache**