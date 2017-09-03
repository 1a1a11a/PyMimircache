# build command: docker build -t 1a1a11a/test1 -f dockerFile .


FROM ubuntu:14.04
MAINTAINER peter.waynechina@gmail.com

ADD testData /mimircache/testData
WORKDIR /mimircache/scripts

# dependency
RUN apt-get -yqq update
RUN apt-get -yqq install python3-pip python3-matplotlib libglib2.0-dev
RUN pip3 install mimircache




# sudo docker run -it --rm -v $(pwd):/mimircache/scripts 1a1a11a/test1 /bin/bash

