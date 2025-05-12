FROM  nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV APP_HOME=/opt/app-root/src
ENV VENV_DIR=$APP_HOME/venv
ENV PATH=$VENV_DIR/bin:$PATH

RUN mkdir -p $APP_HOME
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN  apt update && aptinstall -y tzdata 
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get install -y python3 python3-venv python3-pip git build-essential 

WORKDIR $APP_HOME

RUN python3 -m venv $VENV_DIR
RUN $VENV_DIR/bin/activate
RUN pip install --upgrade pip
RUN pip install terratorch

RUN useradd --uid 10000 --create-home --home-dir /opt/app-root --shell /sbin/nologin appuser

ENV APP_HOME=/opt/app-root/src
ENV VENV_DIR=/opt/app-root/src/venv
ENV PATH=/opt/app-root/src/venv/bin:$PATH

RUN mkdir -p $APP_HOME
RUN chown -R 10000:0 /opt/app-root
RUN chmod -R g+rwX /opt/app-root

USER 10000

CMD ["bash"]
