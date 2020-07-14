## 配置Docker2.0的环境
　　之所以使用docker2.0的原因是在容器中可以使用GPU。
### 配置docker源
```bash
# 更新源
$ sudo apt update

# 启用HTTPS
$ sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

# 添加GPG key
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# 添加稳定版的源
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

### 安装dockerCE

```bash
# 更新源
$ sudo apt update

# 安装Docker CE
$ sudo apt install -y docker-ce
```

### 验证DockerCE

```bash
$ sudo docker run hello-world
```

### 配置docker2源

```bash
# 添加源
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 更新源
$ sudo apt update
```

### 安装nvidia-docker2

```bash
# 安装nvidia-docker2
$ sudo apt install -y nvidia-docker2

# 重启Docker daemon
$ sudo pkill -SIGHUP dockerd
```

### 验证nvidia-docker2

```bash
$ sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
```

