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
        python3 python3-pip python3-setuptools \
        libopenexr-dev && \
    rm -rf /var/lib/apt/lists/*

# ============
# pip packages
# ============
COPY mirrors/aliyun/pip.conf /etc/
RUN python3 -m pip install --upgrade --no-cache-dir pip && \
    python3 -m pip install --upgrade --no-cache-dir setuptools
COPY requirements.txt .
RUN python3 -m pip install --upgrade --no-cache-dir -r requirements.txt

# ===========
# latest apex
# ===========
RUN python3 -m pip install --verbose --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/master.zip

# ==============
# copy all files
# ==============
COPY . .
