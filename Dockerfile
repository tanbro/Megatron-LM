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
        python3 python3-dev python3-pip \
        libopenexr-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# ============
# pip packages
# ============
COPY mirrors/aliyun/pip.conf /etc/
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade setuptools
COPY requirements-*.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade -r requirements-base.txt && \
    python3 -m pip install --verbose --no-cache-dir --upgrade -r requirements-apex.txt

# ==============
# copy all files
# ==============
COPY . .
