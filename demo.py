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
            # Agent('~/Videos/jp/L1-CP-F12_DRIVEWAY - 1280 x 720 - 10fps_20191220_155815.avi', opt, host),
            # Agent('~/Videos/jp/L1-MB-PTZ1_LOADING BAYS - 1280 x 720 - 10fps_20191220_160054.avi', opt, host),
            # Agent('~/Videos/jp/L1-MB-PTZ1_LOADING BAYS - 1280 x 720 - 10fps_20191220_170128.avi', opt, host),
            # Agent('~/Videos/jp/L2-MB-D05-Cityroom 2 - 1280 x 720 - 10fps_20191218_130108.avi', opt, host),

            #VehicleAgent('/home/jeff/Desktop/capitaland_footage/L1-CP-F11_CARPARK BARRIER - 1280 x 720 - 10fps_20191218_080002.avi', opt, host),
            VehicleAgent('/home/pensees/Desktop/capitaland_footage/L1-REC-F03_DROP-OFF POINT - 1280 x 720 - 10fps_20191218_185810.avi', opt, host),
            VehicleAgent('/home/pensees/Desktop/capitaland_footage/L1-CP-F11_CARPARK BARRIER - 1280 x 720 - 10fps_20191218_095417.avi', opt, host, scene = 1),
            PersonAgent('/home/pensees/Desktop/capitaland_footage/L1-CP-F12_DRIVEWAY - 1280 x 720 - 10fps_20191220_155815.avi', host),
            PersonAgent('/home/pensees/Desktop/capitaland_footage/L1-MB-PTZ1_LOADING BAYS - 1280 x 720 - 10fps_20191220_160054.avi', host),
            #VehicleAgent('/home/jeff/Desktop/capitaland_footage/L1-REC-F03_DROP-OFF POINT - 1280 x 720 - 10fps_20191218_194842.avi', opt, host),

            # Agent('../../Videos/MOT16-02_1080p.mp4'),    # 01
            #Agent('../../Videos/100.m4v'), #Agent('../../Videos/101.m4v'),    # CVPR19-02.mp4
            #Agent('../../Videos/126.m4v'),    # 07
        ])

    main()
