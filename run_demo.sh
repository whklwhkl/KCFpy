rm -rf ./output

python demo.py \
--data data/coco.data \
--weights ./models/yolov3.weights \
--cfg cfg/yolov3.cfg \
--nms-thres 0.1 \
--conf-thres 0.2 \
--debug
