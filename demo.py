from scr import Main, Agent


if __name__ == '__main__':
    ips = []

    with open('ipcam.txt') as f:
        for line in f:
            ips += [line.split()[3]]
    # main = Main([Agent('rtsp://admin:888888@{}:10554/udp/av0_0'.format(ip))
    #              for ip in ips])
    main = Main([
        Agent('/home/wanghao/Videos/CVPR19-01.mp4'),
        Agent('/home/wanghao/Videos/CVPR19-02.mp4'),
        Agent('/home/wanghao/Videos/CVPR19-07.mp4'),
    ])
    main()
