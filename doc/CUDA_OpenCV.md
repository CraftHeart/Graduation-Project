# CUDA OpenCV配置

## CUDA配置

## 安装CUDA Toolkit
1. 在官网下载cuda toolkit : [下载](https://developer.nvidia.com/cuda-80-ga2-download-archive)  

2. 切换到下载文件的目录：   
`sudo dpkg -i   cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb`  
`sudo apt-get update`  
`sudo apt-get install cuda`  
注意：如果最后提示uefi secure boot enable的话，重启电脑，进入bios，关闭scure boot即可。  
 
3. 添加环境变量  
`export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}  
sudo ln -s /usr/lib/nvidia-346/libnvcuvid.so /usr/lib/libnvcuvid.so  
sudo ln -s /usr/lib/nvidia-346/libnvcuvid.so.1 /usr/lib/libnvcuvid.so.1`   
此处安装时要根据自己的cuda版本来，最好是自己手打，不要复制粘贴，容易添加环境变量出错。  

4. 编译cuda samples  
进入到samples 文件夹，make -j即可。  
编译成功后，会生成/usr/local/cuda-8.0/samples/bin/x86_64/linux/release文件夹，里面都是编译好的可执行文件，./deviceQuery可以运行测试。

## 编译安装OpenCV
1. 下载Opencv安装包
2. 安装依赖
```
sudo apt-get install libopencv-dev build-essential checkinstall cmake pkg-config yasm libtiff5-dev libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

```
3. 编译OpenCV
```$xslt
cd opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_CUDA=ON -D WITH_NVCUVID=ON -D CUDA_ARCH_BIN="5.0"  ..
```
4. 检查输出，如果输出中的以下部分如图所示，则编译成功  
![检查Opencv安装输出](https://github.com/CraftHeart/Graduation-Project/blob/project/doc/pic/%E6%A3%80%E6%9F%A5Opencv%E5%AE%89%E8%A3%85%E8%BE%93%E5%87%BA.png)  
5. 如果以上步骤正确，则开始安装Opencv
```
make -j8  //建议多线程编译.单线程超慢
sudo make install
```
如果make -j8的时候报错：   
```
_compile_generated_gpu_mat.cu.obj  
nvcc fatal : Unsupported gpu architecture 'compute_11'  
```
cmake的时候可以添加:  
```
-D CUDA_ARCH_BIN="5.0"  
```
这个可以在运行./deviceQuery里面看到  
6. 使得动态链接生效
```$xslt
sudo vim /etc/ld.so.conf
添加 /usr/local/lib
sudo ldconfig -v
```
7. 查看Opencv链接的库
```$xslt
pkg-config --cflags --libs opencv
```
8. 查看Opencv版本
```$xslt
pkg-config --modversion opencv
```
