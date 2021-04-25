docker run -it \
             --gpus all \
             --privileged \
             --env="DISPLAY=$DISPLAY" \
             --env="QT_X11_NO_MITSHM=1" \
             --device /dev/video0:/dev/video0:mwr \
             --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
             --name="stylegan2_test" \
             stylegan2_test:latest
