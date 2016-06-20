.. _installation:

Installation
============
.. warning:: 
	mimircache has been tested on python3.4 and python3.5. Theoretically, all python3 are supported, but if you are using python 3.1~3.3, you might have a hard time install pip, here we suggest using the latest version of python3. 

This part of the documentation covers the installation of mimircache. Mimircache currently has the following dependencies: 
**pkg-config, glib, scipy, numpy, matplotlib**

First Step: Install C Library
----------------------------- 
The first step to using any package management software to install pkg-config and glib. 

For Mac Users: if you don't know how to install these packages, try macports or homebrew, google will help you. 


Second Step: Pip Install mimircache
-----------------------------------
To install mimircache, simply run this command in your terminal of choice::

    $ pip3 install mimircache (you might need sudo) 


Alternative
-------------
Do the following only when you can't install using the methods above, or you want to try the newest version of mimircache, notice that, it might have bugs in newest version, we highly recommend using the stable version from pip3. 

Get the Source Code
+++++++++++++++++++

mimircache is actively developed on GitHub, where the code is 
`always available <https://github.com/1a1a11a/mimircache>`_.

You can clone the public repository::

    $ git clone git://github.com/1a1a11a/mimircache.git

Once you have a copy of the source, you can install it into your site-packages easily::

    $ python3 setup.py install
