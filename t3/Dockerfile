FROM ubuntu
# ...
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*

ADD main.c main.c
ADD ./inputs/arq3.in ./arq3.in

ENTRYPOINT ["bash"]