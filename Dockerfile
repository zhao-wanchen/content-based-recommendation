FROM python:3.8-slim-buster 

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 
RUN apt update
RUN apt install -y python3-pip
RUN apt-get upgrade -y 

COPY * /workspace/

ARG DB_HOST
ARG DB_PSSW
ARG DB_PORT
ARG DB_USER
ARG DB_SCHEMA

ENV DB_HOST "${DB_HOST}"
ENV DB_PSSW "${DB_PSSW}"
ENV DB_PORT "${DB_PORT}"
ENV DB_USER "${DB_USER}"
ENV DB_SCHEMA "${DB_SCHEMA}"

WORKDIR "/workspace"

RUN pip3 install -r /workspace/requirements.txt
RUN python -m nltk.downloader punkt
RUN [ "python", "-c", "import nltk; nltk.download('all')" ]

CMD [ "python3", "main.py"]