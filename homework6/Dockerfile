FROM ubuntu

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /


COPY ./assignment6.sh assignment6.sh
COPY ./baseballscripts.sql baseballscripts.sql

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
    build-essential \
    python3 \
    mysql-client \
    mysql-server \
  && rm -rf /var/lib/apt/lists/*
