FROM continuumio/miniconda:latest

RUN pip install --upgrade pip \
  && pip install --no-cache flask \
  && pip install numpy

WORKDIR /app

COPY ./data_server.py .

EXPOSE 6669

CMD python data_server.py
