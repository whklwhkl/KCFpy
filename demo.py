
from scr import Main, Agent


if __name__ == '__main__':

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
            Agent('../../Videos/100.m4v'),    # 01
            Agent('../../Videos/101.m4v'),    # CVPR19-02.mp4
            Agent('../../Videos/126.m4v'),    # 07
        ])

    main()
