# 配置原生环境

在 [Ubuntu][] `1804 LTS x86_64` 上安装配置可执行该项目的原生环境。

## 硬件要求

支持 NVIDIA [CUDA][] `10.0` 的图形卡设备。

## 操作步骤

建议单独新建一个虚拟环境([venv][])运行该项目的代码。
当然，也可以使用其它的环境隔离方法，如:

- 使用操作系统账户进行隔离
- 使用 [Conda][] 环境进行隔离
- 使用 [docker][] 容器进行隔离

文中不记载其它方法。

1. 安装所需系统软件

   使用 [Apt][] 安装以下系统软件：

   ```bash
   sudo apt install build-essential python3 python3-pip python3-venv
   ```

1. 安装 [CUDA][] `10.0` 和相关软件

   > ❗ **注意**:
   >
   > [PyTorch][] 目前在 [PyPi][] 上最新稳定版默认采用 [CUDA][] `10.0`；在 [Conda][] 上最新稳定版支持的最高 [CUDA][] 版本也是 `10.0`。
   > 所以，我们应安装这个版本的 [CUDA][]

   此处仅记录通过 [Apt][] 在线安装的方法，其它各种安装方式可参考 <https://developer.nvidia.com/cuda-downloads>

   **安装完毕后需要重启计算机**

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
   sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda-10-0
   ```

1. 新建 [Python][] 虚拟环境

   使用 [venv][] 在名为`env`(也可以使用其它名称)的子目录新建虚拟环境，并更新该环境中的 [pip][] 与 [setuptools][] 软件包：

   > 💡 **提示**:
   >
   > 如果同时安装了 [Conda][] 或者其它形式的多个 [Python][] 并存的运行环境，在新建 [venv][] 时，注意不要搞错。
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

   [Apex][] 在安装 [PyTorch][] 之后才可以安装，所以分开两个步骤：

   1. 从 [PyPI][] 安装除 NVIDIA [Apex][] 之外的软件:

      ```bash
      env/bin/pip install -r requirements-base.txt
      ```

   1. 从 [Github](https://github.com/nvidia/apex) 下载 NVIDIA [Apex][] `master` 分支，然后从源代码构建并安装:

      ```bash
      env/bin/pip install -v -r requirements-apex.txt
      ```

------

现在，这个项目可以用该环境下的 [Python][] 执行了！

------

[Ubuntu]: https://www.ubuntu.com/ "Ubuntu is an open source software operating system that runs from the desktop, to the cloud, to all your internet connected things."
[Python]: https://www.python.org/ "Python is a programming language that lets you work quickly and integrate systems more effectively."
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
