# 如何配置运行环境

## 前提要求

- 硬件：

  - `CPU`: `x86_64` 架构
  - `GPU`: 兼容 [CUDA][]`10.0`

- 操作系统：[Ubuntu][] `1804 x86_64`

  任何兼容 [CUDA][]`10.0` 的 `Linux x86_64` 操作系统都可运行该项目，但本文仅以 [Ubuntu][]`1804 x86_64` 为例。
  详情请参考 <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements>

## 环境隔离

强烈建议为该项目准备一个隔离的运行环境，它可以是：

- [venv][] 虚拟环境
- [Conda][] 环境
- [Docker][] 容器

## 系统全局 / venv 环境

1. 安装所需系统软件

   使用 [Apt][] 安装以下系统软件：

   ```bash
   sudo apt install wget build-essential libopenexr-dev python3 python3-dev python3-pip python3-venv
   ```

1. 安装 [CUDA][] `10.0`

   > ❗ **注意**:
   >
   > [PyTorch][] 目前在 [PyPi][] 上最新稳定版默认采用 [CUDA][] `10.0`；在 [Conda][] 上最新稳定版支持的最高 [CUDA][] 版本也是 `10.0`。
   > 所以，我们应安装这个版本的 [CUDA][]

   此处仅记录通过 [Apt][] 在线安装的方法，其它各种安装方式可参考 <https://developer.nvidia.com/cuda-downloads>

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
   sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda-10-0
   ```

   **该步骤安装完毕后需重启计算机**

1. 安装 `cuDNN for CUDA 10.0`

   使用浏览器访问 <https://developer.nvidia.com/cudnn> （需要事先注册[NVIDIA][]账号），依次点击超链接：

   1. `cuDNN for CUDA 10.0` 最新稳定版
   1. `Library for Windows, Mac, Linux, Ubuntu and RedHat/Centos(x86_64architectures )`

   选择下载这两个安装包：

   - cuDNN Runtime Library for Ubuntu18.04 (Deb)
   - cuDNN Developer Library for Ubuntu18.04 (Deb)

   下载完成后，使用 [Apt][] 安装下载的两个安装包文件，例如：

   ```bash
   cd ~/下载
   sudo apt install ./libcudnn7_7.6.4.38-1+cuda10.0_amd64.deb
   sudo apt install ./libcudnn7-dev_7.6.4.38-1+cuda10.0_amd64.deb
   ```

1. 新建 [Python][] 虚拟环境(*可选*/*推荐*)

   使用 [venv][] 在名为`env`(也可以使用其它名称)的子目录新建虚拟环境，并更新该环境中的 [pip][] 与 [setuptools][] 软件包：

   > 💡 **提示**:
   >
   > 如有 [Conda][] 或者其它多个 [Python][] 并存的情况，在新建 [venv][] 时，注意不要搞错。
   > 可以使用 `which` 命令检查，如：
   >
   > ```bash
   > $ which python3
   > /usr/bin/python3
   > ```

   ```bash
   cd /path/of/this/project
   python3 -m venv env
   env/bin/pip install --upgrade pip
   env/bin/pip install --upgrade setuptools
   ```

1. 安装所需 [Python][] 软件包

   > 💡 **提示**:
   >
   > 如有 [Conda][] 或者其它多个 [Python][] 并存的情况，在使用 [pip][] 安装软件包时，注意不要搞错。
   > 可以这样检查：
   >
   > ```bash
   > $ pip3 --version
   > pip 19.2.3 from /home/xxx/.local/lib/python3.6/site-packages/pip (python 3.6)
   > ```

   [Apex][] 在 [PyTorch][] 安装完毕之后才可以安装，所以分开两个步骤：

   1. 从 [PyPI][] 安装除 [Apex][] 之外的软件:

      - 如果采用系统全局环境：

        ```bash
        cd /path/of/this/project
        sudo -H pip3 install --upgrade -r requirements-base.txt
        ```

      - 如果采用 [venv][] 虚拟环境：

        ```bash
        cd /path/of/this/project
        env/bin/pip install --upgrade -r requirements-base.txt
        ```

   1. 从 [Github](https://github.com/nvidia/apex) 下载 [Apex][] 的 `master` 分支，然后从源代码构建并安装:

      - 如果采用系统全局环境：

        ```bash
        cd /path/of/this/project
        sudo -H pip3 install -v --upgrade -r requirements-apex.txt
        ```

      - 如果采用 [venv][] 虚拟环境：

        ```bash
        cd /path/of/this/project
        env/bin/pip install -v --upgrade -r requirements-apex.txt
        ```

至此，环境已配置完毕：

- 如果在系统全局配置了上述环境，可使用系统级的 `python3` 调用项目脚本，如：

  ```bash
  python3 pretrain_gpt2.py
  ```

- 如果在虚拟目录配置了上述环境，可使用该虚拟目录中的 `python` 调用项目脚本，如：

  ```bash
  env/bin/python pretrain_gpt2.py
  ```

> ‼ **注意**:
>
> 小心多个 [Python][] 运行时并存的情况，不要搞错路径。

## Conda 环境

我们可以将 [CUDA][] 等软件安装在 [Conda][] 环境中，而不必安装到操作系统(仍需要在系统中安装`Driver`)。

1. 按照 [Conda][] 官方文档的指导，安装并配置 `Anaconda` 或者 `Miniconda`。

   - 如安装 `Anaconda`，请参考 <https://docs.anaconda.com/anaconda/install/linux/>
   - 如安装 `Miniconda`，请参考 <https://conda.io/projects/conda/user-guide/install/linux.html>

1. 安装 [NVIDIA][] `Driver`。

   为了配合 [CUDA][]`10.0`，建议安装`418`版本：

   ```bash
   sudo add-apt-repository restricted
   sudo apt update
   sudo apt install nvidia-driver-418
   ```

   **该步骤安装完毕后需重启计算机**

1. 创建 [Conda][] 环境。

   该项目提供了 [Conda][] 配置文件 `environment.yml`，我们可以直接由这个配置文件创建名为 `Megatron-LM` 的 [Conda][] 环境，并同时在该环境中安装所需软件包（除了 [Apex][]）：

   ```bash
   cd /path/of/this/project
   conda env update -f environment.yml
   ```

1. 安装 [Apex][]

   [Apex][] 在 [PyTorch][] 安装后方可安装，且没有官方 [Conda][] 源，所以我们只能在环境创建后，手动调用 [pip][] 从源代码进行构建和安装：

   ```bash
   cd /path/of/this/project
   conda activate Megatron-LM
   pip install -v -r requirements-apex.txt
   ```

现在，我们已经在名为 `Megatron-LM` 的 [Conda][] 环境中配置了该项目所需的 [Python][] 运行时。激活环境即可使用，如：

```bash
cd /path/of/this/project
conda activate Megatron-LM
python pretrain_gpt2.py
```

## Docker 容器

我们可以将 [CUDA][] 等软件，以及这个项目本身，全部安装在 [Docker][] 镜像中，而不必安装到操作系统(仍需要在系统中安装`Driver`与 [NVIDIA Container Toolkit][])。

项目根目录存放有 [Docker][] 配置文件`Dockerfile`。另外，这个项目的派生来源([NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM))也在`docker`目录下提供了 [Docker][] 配置文件。此处以我们自己的为准。

1. 安装 [Docker][] `19.03` 或以上版本。

   由于 [NVIDIA Container Toolkit][] 对 [Docker][] 的版本要求，我们无法直接使用 [Ubuntu][] `1804` 官方提供的较低版本 [Docker][]，而是要安装较新的稳定版。

   建议参考 <https://docs.docker.com/install/linux/docker-ce/ubuntu/> 安装满足版本要求的 [Docker][]`CE`。此处亦作简单记录：

   ```bash
   sudo apt-get remove docker docker-engine docker.io containerd runc
   sudo apt-get update
   sudo apt-get install \
       apt-transport-https \
       ca-certificates \
       curl \
       gnupg-agent \
       software-properties-common
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   sudo add-apt-repository \
      "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) \
      stable"
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

2. 安装 [NVIDIA][] `Driver`。

   为了配合 [CUDA][]`10.0`，建议安装`418`版本：

   ```bash
   sudo add-apt-repository restricted
   sudo apt update
   sudo apt install nvidia-driver-418
   ```

   **该步骤安装完毕后需重启计算机**

3. 安装 [NVIDIA Container Toolkit][]

   它为 [Docker][] 容器提供`GPU`加速功能，它的安装和使用方法详见其网站。
   本例中，我们这样安装：

   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. 构建 [Docker][] 镜像

   ```bash
   cd /path/of/this/project
   sudo docker build --no-cache .
   ```

   本项目的脚本复制到了镜像的 `/root` 目录。

构建完毕后，我们可以通过 [Docker][] 容器运行这个镜像中的项目脚本，如（假设该镜像的 `ID` 是 `123456789abc`）：

```bash
sudo docker run --gpus=all -it --rm -e LANG=C.UTF-8 -v /path/of/data:/root/data -v /path/of/checkpoints:/root/checkpoints 123456789abc python3 generate_samples.py --load checkpoints/model/dir --tokenizer-type SentencePieceTokenizer --tokenizer-path data/path/of/spm.model
```

------

[NVIDIA]: https://www.nvidia.com/
[Ubuntu]: https://www.ubuntu.com/ "Ubuntu is an open source software operating system that runs from the desktop, to the cloud, to all your internet connected things."
[Apt]: https://help.ubuntu.com/lts/serverguide/apt.html "The apt command is a powerful command-line tool, which works with Ubuntu's Advanced Packaging Tool (APT) performing such functions as installation of new software packages, upgrade of existing software packages, updating of the package list index, and even upgrading the entire Ubuntu system."
[Docker]: https://www.docker.com/ "Docker: The Modern Platform for High-Velocity Innovation"
[Python]: https://www.python.org/ "Python is a programming language that lets you work quickly and integrate systems more effectively."
[PyPI]: https://pypi.org/ "Find, install and publish Python packages with the Python Package Index"
[pip]: https://packaging.python.org/key_projects/#pip "A tool for installing Python packages."
[venv]: https://packaging.python.org/key_projects/#venv "A package in the Python Standard Library (starting with Python 3.3) for creating Virtual Environments."
[setuptools]: https://packaging.python.org/key_projects/#easy-install "setuptools (which includes easy_install) is a collection of enhancements to the Python distutils that allow you to more easily build and distribute Python distributions, especially ones that have dependencies on other packages."
[wheel]: https://packaging.python.org/key_projects/#wheel "bdist_wheel setuptools extension for creating wheel distributions."
[Conda]: https://packaging.python.org/key_projects/#conda "conda is the package management tool for Anaconda Python installations."
[CUDA]: https://developer.nvidia.com/cuda-toolkit "The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. "
[NVIDIA Container Toolkit]: https://github.com/NVIDIA/nvidia-docker "The NVIDIA Container Toolkit allows users to build and run GPU accelerated Docker containers. "
[PyTorch]: https://pytorch.org/ "An open source machine learning framework that accelerates the path from research prototyping to production deployment."
[Apex]: https://nvidia.github.io/apex/ "Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training."
