# ===========
# base images
# ===========
FROM nvidia/cuda:10.1-devel


WORKDIR /root


# ==============
# copy all files
# ==============
COPY . .

# ===============
# system packages
# ===============
RUN cp -f mirrors/aliyun/sources.list /etc/apt/ && \
    apt-get update && \
    apt-get install -y python3 python3-dev python3-pip libopenexr-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# ============
# pip packages
# ============
RUN cp -f mirrors/aliyun/pip.conf /etc/ && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade setuptools && \
    python3 -m pip install --no-cache-dir --upgrade -r requirements/base.txt && \
    python3 -m pip install --verbose --no-cache-dir --upgrade -r requirements/apex.txt && \
    rm -rf /tmp/pip-*
