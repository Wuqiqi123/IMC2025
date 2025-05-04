FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Docker container for IMC with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git

WORKDIR /workspace

COPY requirements.txt  /workspace/requirements.txt 
RUN pip install -r requirements.txt --index https://mirrors.ustc.edu.cn/pypi/web/simple/
