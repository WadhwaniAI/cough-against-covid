# build the image required for setting up the repository
# Example run: 
# $ docker build -t wadhwaniai/cough-against-covid:py3-1.0 .
# Creates a docker image with desired dependencies

# base image
FROM nvcr.io/nvidia/pytorch:20.01-py3

RUN apt-get update && apt-get install -y \
    tmux \
    p7zip-full p7zip-rar \
    aufs-tools \
    automake \
    build-essential \
    curl \
    dpkg-sig \
    libcap-dev \
    libsqlite3-dev \
    mercurial \
    virtualenv \
    reprepro \
    ruby1.9.1 && rm -rf /var/lib/apt/lists/*

# change working directory to /
WORKDIR /

# set the PYTHONPATH required for using the repository
ENV PYTHONPATH /workspace/cough-against-covid

# set actual working directory
WORKDIR /workspace/cough-against-covid

# copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN cat requirements.txt | xargs -n 1 pip install | while read line; do echo $line; done;
RUN pip install git+https://github.com/detly/gammatone.git
