FROM ubuntu

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
    build-essential \
    python3 \
    mariadb-client \
  && rm -rf /var/lib/apt/lists/*

COPY ./baseballscripts.sql .
COPY ./assignment6.sh .
COPY baseball.sql .

CMD ./assignment6.sh
