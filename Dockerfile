FROM  nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip3 install --upgrade pip && pip3 install terratorch
