from scr import Main, Agent
# TODO: person_agent, vehicle_agent


if __name__ == '__main__':
    host = '192.168.20.191'
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
            Agent('~/Videos/jp/L1-CP-F12_DRIVEWAY - 1280 x 720 - 10fps_20191220_155815.avi', host),
            Agent('~/Videos/jp/L1-MB-PTZ1_LOADING BAYS - 1280 x 720 - 10fps_20191220_160054.avi', host),
            Agent('~/Videos/jp/L1-MB-PTZ1_LOADING BAYS - 1280 x 720 - 10fps_20191220_170128.avi', host),
            Agent('~/Videos/jp/L2-MB-D05-Cityroom 2 - 1280 x 720 - 10fps_20191218_130108.avi', host),
            # Agent('../../Videos/MOT16-02_1080p.mp4'),    # 01
            #Agent('../../Videos/100.m4v'), #Agent('../../Videos/101.m4v'),    # CVPR19-02.mp4
            #Agent('../../Videos/126.m4v'),    # 07
        ])

    main()
