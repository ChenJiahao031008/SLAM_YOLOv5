## SLAM-YOLOv5

做物体SLAM的时候本来想找到一个和SLAM结合的，并且是c++版本的目标检测系统，github上面一搜，不是很多，大部分都是离线检测，再不然就是ROS版本或者Python版本的YOLO（不是说不行，但是个人感觉速度很奇怪，速度也不快），满足我要求的几乎没有，因此干脆花了点时间自己东拼西凑一个出来。

代码链接：https://github.com/ChenJiahao031008/SLAM-YOLOv5.git

### 环境配置

由于深度学习不可避免的涉及到显卡安装等各类问题，这里把我自己的配置罗列下（安装过程没啥困难，也比较基础，就不展开了），因为自己刚配的电脑，比较新，大家可以适当放低配置什么的。

+ 系统 Ubuntu16.04

+ RTX 3060 Laptop显卡 Driver 470.42.01

+ CUDA 11.1（显卡太新以至于11以下的不兼容）

+ cudnn 8.1.0.77

+ OpenCV 3.4.4 （无cuda版本，因为不支持cuda11）

+ LibTorch： libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111 

+ （可选）pytorch ：1.9.0-cu111版本

### 运行实例

注意，运行前修改`CMakeLists.txt`文件中libtorch库的位置（或者CMAKE时候指定），修改`YOLOv5Detector.cc`中的硬地址等

```bash
mkdir build && cd build
cmake ..
make -j10
./Examples/RGB-D/rgbd_tum  Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml  Dataset/rgbd_dataset_freiburg1_desk Dataset/rgbd_dataset_freiburg1_desk/associate.txt
```

### 改进思路

YOLO这块主要参考了https://github.com/Nebula4869/YOLOv5-LibTorch。

核心思想比较简单，就是把Python版本的训练好的.pt文件转换ONNX推理框架，生成TorchScript文件，这里主要参考了代码：https://github.com/ultralytics/yolov5/tree/v5.0

运行指令：

```bash
export PYTHONPATH="$PWD"
# 在有pytorch的环境下 
python models/export.py --device 0
```

其中，需要下载`yolov5s.pt`在主文件夹下（不放的话也会自动下），在工程目录下生成`yolov5s.torchscript.pt`，然后讲生成的TorchScript文件放到SLAM框架下使用。

在SLAM中，单线程加入YOLO是比较简单的，只需要对CMakeLists文件做一些改动即可。注意，进行推理可能用到C++14标准，需要切换标志符。同时，这也会对SLAM系统产生一点影响（不过问题不大）

提示：关闭警告可以在CMakeLists中加入：

```cmake
#添加的部分，关闭警告
add_definitions(-w)
```

### 多线程配置

为了提高性能，可以对YOLO另外开一个线程，这块涉及到了ORB-SLAM2的多线程系统设计，简单说一下：

多线程入口，仿照其他线程设计即可。看代码

```c++
#ifdef USE_YOLO_DETECTOR
    std::cout << "[INFO] USE_YOLO_DETECTOR." << std::endl;
    mpDetector = new YOLOv5Detector(mpFrameDrawer);
    mptDetector = new thread(&ORB_SLAM2::YOLOv5Detector::Run, mpDetector);
#endif
```

注意，这里仅仅是把YOLOv5嵌入了SLAM系统，在viewer线程中展示出来，并没有做任何额外的处理。

因此多线程的配合是比较简单的，`YOLOv5Detector`类只和`FrameDrawer`交互，从后者中读取需要处理的图像，并传回结果图像（中间过程有一些互锁环节）。而`FrameDrawer`会在`viewer`线程中被调用和结果展示。

如果大家感兴趣可以继续扩展，就写到这里了。

