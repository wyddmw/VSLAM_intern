### 安装并使用AirSim

　　为了能够更好地管理服务器上的环境，AirSim的使用将通过Docker构建镜像来完成。AirSim官方给出了两种不同的方法来进行构建，一种是通过二进制文件的方式——测试失败了，会报音频设备检测不到的错误，可能是docker容器中没有安装对应设备的原因。所以选择使用通过源码编译的方式进行安装。https://microsoft.github.io/AirSim/docker_ubuntu/#requirements链接中给出了基本的教程。

　　首先我们要安装nvidia-docker2，可以更好地支持在容器中使用GPU，与宿主机上的CUDA环境是一致的，安装nvidia-docker2的方法可以参考链接<https://docs.docker.com/engine/install/ubuntu/>。完成安装docker之后需要安装ue4-docker，安装的说明参考链接<https://docs.adamrehn.com/ue4-docker/configuration/configuring-linux#step-3-install-ue4-docker>。

#### 注意：经过测试，4.19.2的版本在编译的时候会出现错误，原因是版本的问题，经过测试4.21.1的版本是可以安装成功的

```bash
$sudo pip3 install ue4-docker
$sudo ue4-docker setup
$ue4-docker build 4.21.1 --cuda=10.2 --no-full # 根据对应cuda的版本进行修改
```

　　需要注意的是，在进行docker镜像下载的时候会出现速度过慢的情况，解决的方法是通过阿里云中的镜像容器服务获取镜像加速器：

```bash
$ sudo mkdir -p /etc/docker
$ sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://02dn6dgi.mirror.aliyuncs.com"]		# 从自己的阿里云容器镜像服务中获取加速器地址
}
EOF
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```



## Build AirSim inside UE4 docker container:

　　AirSim的docker镜像是建立在unreal engine镜像的基础之上的。

## 将制作好的docker镜像上传到docker hub上

　　在本地上先登录到docker上：

```bash
$ sudo docker login
$ # 然后输入我们dockerhub的账号还有密码就可以实现登录
$ sudo docker push
```

