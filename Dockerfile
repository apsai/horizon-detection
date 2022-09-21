FROM ubuntu:20.04

WORKDIR /app

# don't bother me
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt upgrade -y

RUN apt install -y \
    vim \
    gcc \
    wget \
    pkg-config \
    libcairo2-dev \
    software-properties-common \
    python3-dev \
    python3-pip \
    python3-opencv \
    libopencv-dev

# upgrade numpy directly
RUN pip3 install --upgrade numpy==1.23.3

COPY utils/ /app/utils/
COPY main.py /app/
COPY README.md /app/

CMD ["python3", "-u", "main.py"]

