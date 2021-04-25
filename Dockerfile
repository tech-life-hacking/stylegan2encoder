FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && apt-get install -y git wget \
    libffi-dev \
    libssl-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev

RUN pip install -U pip

RUN pip install opencv-python opencv-contrib-python matplotlib tqdm pandas matplotlib
RUN pip install scipy==1.1.0
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install keras==2.2.4
RUN pip install cmake
RUN pip install dlib imutils

RUN mkdir /home/stylegan2encoder

RUN git clone https://github.com/tech-life-hacking/stylegan2encoder.git /home/stylegan2encoder
RUN wget -P /home/stylegan2encoder http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
RUN bzip2 -d /home/stylegan2encoder/shape_predictor_68_face_landmarks.dat.bz2