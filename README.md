# KCF tracker in Python
Intended to be used with other apps to do person re-identification

## prerequisites
- person detection service, suppose the image's name is `per_det`
- person feature extraction service, suppose the image's name is `fea_ext`

### ports assumptions:
  - `per_det`:  6666
  - `fea_ext`:  6667
  - `reid`:     6669

## setting up

1. `make` to build the data server's docker image `reid`

2. `docker-compose up -d` launch services

## run

`python demo.py` launch client demo
