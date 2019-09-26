video_folder=$1

for i in  `find $video_folder -type f|grep mp4`;do
  echo $i >> log.txt
  PYTHONPATH=. python scripts/parse_video.py \
    -v $i \
    -p 6666 \
    -f 24
done
