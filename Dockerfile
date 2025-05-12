FROM ubuntu:24.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /opt/app-root/src
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN  apt update && apt install -y tzdata 
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get install -y python3 python3-venv python3-pip git build-essential 
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /opt/app-root/src

# Create venv and install terratorch
RUN python3 -m venv /opt/app-root/src/venv
RUN chmod 755 /opt/app-root/src/venv/bin/activate
RUN . /opt/app-root/src/venv/bin/activate && pip install  --no-cache-dir --upgrade pip && pip install  --no-cache-dir terratorch
RUN pip cache purge

RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*


RUN useradd --uid 10000 --create-home --home-dir /opt/app-root --shell /sbin/nologin appuser

ENV APP_HOME=/opt/app-root/src
ENV VENV_DIR=/opt/app-root/src/venv
ENV PATH=/opt/app-root/src/venv/bin:$PATH

RUN chown -R 10000:0 /opt/app-root
RUN chmod -R g+rwX /opt/app-root


FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

COPY --from=builder /opt/app-root/src/venv /opt/app-root/src/venv

USER 10000

CMD ["bash"]
