FROM python:3.10

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
    build-essential \
    python3 \
    mariadb-client \
    python3-pip \
    python3-dev \
    libmariadb-dev \
  && rm -rf /var/lib/apt/lists/*

COPY baseball.sql .
COPY ./requirements.txt .
RUN pip3 install --upgrade mysql-connector-python
RUN python3 -m pip install --upgrade pip
RUN pip3 install --compile --no-cache-dir -r requirements.txt

COPY ./finalbaseball.sql .
COPY ./final.sh .
COPY final/main.py .
COPY final/predfeatures.py .



RUN chmod +x ./final.sh .
ENTRYPOINT ["./final.sh"]
CMD ./final.sh

