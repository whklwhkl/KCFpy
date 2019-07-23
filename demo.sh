for i in \
'rtsp://admin:888888@192.168.1.101:10554/udp/av0_0' \
'rtsp://admin:888888@192.168.1.102:10554/udp/av0_0' \
'rtsp://admin:888888@192.168.1.100:10554/udp/av0_0' \
'rtsp://admin:888888@192.168.1.126:10554/udp/av0_0';
do
    python client.py "$i" &
done
