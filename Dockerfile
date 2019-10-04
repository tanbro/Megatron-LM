# ===========
# base images
# ===========
FROM nvidia/cuda:10.0-devel


WORKDIR /root


# ===============
# system packages
# ===============
COPY mirrors/aliyun/sources.list /etc/apt/
RUN apt-get update && \
    apt-get install -y \
        python3 python3-dev python3-pip python3-setuptools \
        git \
        libopenexr-dev && \
    rm -rf /var/lib/apt/lists/*

# ============
# pip packages
# ============
COPY mirrors/aliyun/pip.conf /etc/
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade setuptools
COPY requirements.txt .
RUN python3 -m pip install --ignore-installed --upgrade -r requirements.txt

# ===========
# latest apex
# ===========
RUN python3 -m pip install --verbose --upgrade --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache-dir https://github.com/NVIDIA/apex/archive/master.zip

# ==============
# copy all files
# ==============
COPY . .
