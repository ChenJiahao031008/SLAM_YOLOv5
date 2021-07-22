#ifndef YOLOV5DETECTED_H
#define YOLOV5DETECTED_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <thread>
#include <unistd.h>
#include <mutex>

#include "FrameDrawer.h"

namespace ORB_SLAM2
{

class FrameDrawer;

class YOLOv5Detector
{
public:
    struct Object
    {
       int id;
       int w, h;
       cv::Point2i corner;
       float score = 1.0;

       Object(int id_, int x, int y, int w_, int h_, float score_) : id(id_), corner(cv::Point2i(x,y)), w(w_), h(h_), score(score_){};

       Object(){};

       void print() { std::cout << "[INFO] Object " << id << ":  " << corner << "\t" << w << "\t" << h << std::endl; };
    };

    cv::Mat mImage;

public:
    YOLOv5Detector(FrameDrawer *pFrameDrawer);

    std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

    void Run();

    void RequestFinish();

    void RequestStop();

    bool isFinished();

    bool isStopped();

    void Release();

    bool Stop();


private:
    std::vector<std::string> classnames;

    torch::jit::script::Module module;

    std::vector<Object> vObject;

    bool CheckFinish();
    void SetFinish();

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

    FrameDrawer *mpFrameDrawer;

    std::mutex mMutex;
};


}
#endif
