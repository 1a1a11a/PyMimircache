FROM ubuntu:14.04

ADD ./ /opt/mimircache
WORKDIR /opt/mimircache

# dependency
RUN apt-get -yqq update
RUN apt-get -yqq install python3-pip python3-matplotlib libglib2.0-dev
RUN pip3 install -r requirements.txt
#RUN export PYTHONPATH=/opt/mimircache/
RUN python3 setup.py install


CMD [ "python3", "test/test_reader.py" ]
WORKDIR /opt/mimircache/user/
