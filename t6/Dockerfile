FROM ubuntu
# ...
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs make vim && \
    rm -rf /var/lib/apt/lists/*

ADD md5.c md5.c
ADD md5.h md5.h
ADD md5_bmark.c md5_bmark.c
ADD md5_bmark_parallel.c md5_bmark_parallel.c
ADD md5_bmark.h md5_bmark.h
ADD Makefile Makefile
ADD ./inputs/* ./

ENTRYPOINT ["bash"]