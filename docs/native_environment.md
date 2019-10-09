# 配置原生环境

在 Ubuntu 1804 x86_64 上配置该项目的原生环境。

## 硬件要求

支持 NVIDIA [CUDA][] `10.0` 的图形卡设备。

## 操作步骤

建议单独新建一个虚拟环境([venv][])运行该项目的代码。
当然，也可以使用其它的环境隔离方法，如:

- 操作系统用户
- [Conda][]
- [docker][]

其它方法本文不记载。

1. 安装系统软件

   ```bash
   sudo apt install build-essential python3 python3-dev python3-pip python3-venv
   ```

1. 安装 [CUDA] 和相关软件:

   此处仅记录通过 [Apt][] 在线安装的方法，具体可参考 <https://developer.nvidia.com/cuda-downloads>

   **安装完毕后需要重启**

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
   sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda-10-0
   ```

1. 新建虚拟环境

   使用 [venv][] 在名为`env`(也可以使用其它名称)的子目录新建虚拟环境，并更新环境中的 [pip][], [setuptools][]

   > 💡 **提示**:
   >
   > 如果同时安装了 [Conda][] 或者多个 Python 版本，在新建 [venv][] 时，不要搞错。可以使用 `which` 命令检查，如：
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

1. 安装 [PyPI][] 软件

   [Apex][] 在安装 [PyTorch][] 之后才可以安装，所以分开两个步骤：

   1. 安装除 NVIDIA [Apex][] 之外的 [PyPI][] 软件

      ```bash
      env/bin/pip install -r requirements-base.txt
      ```

   1. 从 Github 下载代码 安装 NVIDIA [Apex][]

      ```bash
      env/bin/pip install -v -r requirements-apex.txt
      ```

现在，这个项目可以用该环境下的 `python` 执行了！

------

[PyPI]: https://pypi.org/ "Find, install and publish Python packages with the Python Package Index"
[pip]: https://packaging.python.org/key_projects/#pip "A tool for installing Python packages."
[venv]: https://packaging.python.org/key_projects/#venv "A package in the Python Standard Library (starting with Python 3.3) for creating Virtual Environments."
[setuptools]: https://packaging.python.org/key_projects/#easy-install "setuptools (which includes easy_install) is a collection of enhancements to the Python distutils that allow you to more easily build and distribute Python distributions, especially ones that have dependencies on other packages."
[wheel]: https://packaging.python.org/key_projects/#wheel "bdist_wheel setuptools extension for creating wheel distributions."
[Conda]: https://packaging.python.org/key_projects/#conda "conda is the package management tool for Anaconda Python installations."
[Apex]: https://nvidia.github.io/apex/ "Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training."
[CUDA]: https://developer.nvidia.com/cuda-toolkit "The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. "
[PyTorch]: https://pytorch.org/ "An open source machine learning framework that accelerates the path from research prototyping to production deployment."
[Apt]: https://help.ubuntu.com/lts/serverguide/apt.html "The apt command is a powerful command-line tool, which works with Ubuntu's Advanced Packaging Tool (APT) performing such functions as installation of new software packages, upgrade of existing software packages, updating of the package list index, and even upgrading the entire Ubuntu system."
[docker]: https://www.docker.com/ "Docker: The Modern Platform for High-Velocity Innovation"
