# build command: docker build -t 1a1a11a/mimircache:env -f dockerfileEnv .
# push command: docker push 1a1a11a/mimircache

# the difference between this file and Dockerfile is that this one only contains the environment for using mimircache,
# you need to install mimircache by yourself, the other one contains mimircache

FROM ubuntu:14.04
LABEL maintainer="peter.waynechina@gmail.com"

ADD testData /mimircache/testData
WORKDIR /mimircache/scripts

# dependency
RUN apt-get -yqq update
RUN apt-get -yqq install python3-pip python3-matplotlib libglib2.0-dev




# sudo docker run -it --rm -v $(pwd):/mimircache/scripts 1a1a11a/mimircache:env /bin/bash

