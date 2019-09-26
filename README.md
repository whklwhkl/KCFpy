# KCF tracker in Python
Intended to be used with other apps to do person re-identification

## prerequisites
- person detection service, suppose the image's name is `per_det`
- person feature extraction service, suppose the image's name is `fea_ext`

### ports:
  Defined in `src/agent.py`, can be changed on demand.
  - `per_det`:  6666
  - `fea_ext`:  6667
  - `reid`:     6669

## setting up (optional)
If we do not have the services ready, we can launch them on the local host.

1. `make` to build the data server's docker image `reid`

2. `docker-compose up -d` launch services on local host

## run

`python demo.py` launch client demo

# Collect tracked objects from Videos

```bash
PYTHONPATH=. python scripts/parse_video.py \
  -v ${YOUR_VIDEO_PATH} \
  -i ${IP_ADDRESS_OF_DETECTION_SERVICE} \
  -p ${PORT_OF_DETECTION_SERVICE} \
  -f ${DETECTION_PERIOD} \
  -o ${OUTPUT_FOLDER} \
  -s #[optional] play video wile parsing
```
