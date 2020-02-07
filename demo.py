from scr import Main, Agent, PersonAgent
from scr.agents.vehicle import VehicleAgent
import argparse
# TODO: person_agent, vehicle_agent


if __name__ == '__main__':
    #Parse Args for YOLOv3 vehicle detector
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')

    #Debug mode, shows track boxes and worker errors
    parser.add_argument('--debug', action='store_true', help='Whether to launch debug mode')
    parser.add_argument('--record', action='store_true', help='Whether to record video (1st 5 mins)')

    opt = parser.parse_args()

    host = 'localhost'

    if 0:
        ips = []
        with open('ipcam.txt') as f:
            for line in f:
                ips += [line.split()[3]]
        main = Main([Agent('rtsp://admin:888888@{}:10554/udp/av0_0'.format(ip))
                     for ip in ips])
    else:
        print('videos')
        main = Main([
            #RTSP Stream
            #VehicleAgent('rtsp://admin:Admin1234@192.168.66.40:554/cam/realmonitor?channel=1&subtype=0', opt, host),
            #VehicleAgent('rtsp://admin:Admin1234@192.168.66.21:554/cam/realmonitor?channel=1&subtype=0', opt, host, scene = 1),
            #PersonAgent('rtsp://admin:Admin1234@192.168.66.22:554/cam/realmonitor?channel=1&subtype=1', host),
            #PersonAgent('rtsp://admin:Admin1234@192.168.66.69:554/cam/realmonitor?channel=1&subtype=1', host),

            #Demo videos
            VehicleAgent('~/Videos/jp/L1-REC-F03_DROP-OFF POINT - 1280 x 720 - 10fps_20191218_185810.avi', opt, host),
            VehicleAgent('/home/jeff/Videos/cut_test.m4v', opt, host, scene = 1),
            #PersonAgent('~/Videos/jp/L1-CP-F12_DRIVEWAY - 1280 x 720 - 10fps_20191220_155815.avi', host),
            PersonAgent('~/Videos/jp/L1-CP-F12_DRIVEWAY - 1280 x 720 - 10fps_20191220_155815.avi', host),
            PersonAgent('~/Videos/jp/L1-MB-PTZ1_LOADING BAYS - 1280 x 720 - 10fps_20191220_160054.avi', host),
        ], record = opt.record)

    main()
