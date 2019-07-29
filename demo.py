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
            Agent('../../Videos/CVPR19-02.mp4'),    # 01
            Agent('../../Videos/CVPR19-02.mp4'),    # 02
            Agent('../../Videos/CVPR19-02.mp4'),    # 07
        ])

    main()
