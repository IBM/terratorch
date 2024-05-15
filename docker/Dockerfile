FROM pytorch/pytorch:latest

RUN apt-get update -y
RUN apt-get install -y vim

COPY ./ /terratorch/

WORKDIR /terratorch

RUN pip install -r requirements.txt
RUN pip install .

RUN pip cache purge
