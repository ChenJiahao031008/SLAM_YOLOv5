#include "YOLOv5Detector.h"


namespace ORB_SLAM2
{

YOLOv5Detector::YOLOv5Detector(FrameDrawer *pFrameDrawer):
mpFrameDrawer(pFrameDrawer),mbFinishRequested(false), mbFinished(true),
mbStopped(true), mbStopRequested(false)
{
    module = torch::jit::load("/home/chen/桌面/SLAM/ObjectSLAM/YOLOv5-LibTorch-SLAM融合/yolov5s.torchscript.pt");

    std::ifstream f("/home/chen/桌面/SLAM/ObjectSLAM/YOLOv5-LibTorch-SLAM融合/coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        classnames.push_back(name);
    }
    std::cout << "[INFO] YOLOv5 Detector Config Loaded. " << std::endl;
}

void YOLOv5Detector::Run(){
    mbFinished = false;
    mbStopped = false;

    std::cout << "[INFO] START YOLOv5 Detector .... " << std::endl;
    while(1){
        while ( mpFrameDrawer->CheckImageStatus() == false ){}
        mImage = mpFrameDrawer->GetFrame();
        cv::cvtColor(mImage, mImage, CV_GRAY2BGR);
        cv::Mat img;
        // Preparing input tensor
        cv::resize(mImage, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
        imgTensor = imgTensor.permute({2,0,1});
        imgTensor = imgTensor.toType(torch::kFloat);
        imgTensor = imgTensor.div(255);
        imgTensor = imgTensor.unsqueeze(0);

        // preds: [?, 15120, 9]
        torch::Tensor preds = module.forward({imgTensor}).toTuple()->elements()[0].toTensor();
        std::vector<torch::Tensor> dets = non_max_suppression(preds, 0.4, 0.5);

        // cv::Mat showImage = mImage.clone();
        if (dets.size() > 0)
        {
            // Visualize result
            for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
            {
                float left = dets[0][i][0].item().toFloat() * mImage.cols / 640;
                float top = dets[0][i][1].item().toFloat() * mImage.rows / 384;
                float right = dets[0][i][2].item().toFloat() * mImage.cols / 640;
                float bottom = dets[0][i][3].item().toFloat() * mImage.rows / 384;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();
                vObject.emplace_back(classID, left, top, (right - left), (bottom - top), score);
                cv::rectangle(mImage, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);

                cv::putText(mImage,
                            classnames[classID] + ": " + cv::format("%.2f", score),
                            cv::Point(left, top),
                            cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 255, 0), 2);
            }

        }
        // else{
        //     std::cout << "NO DETECTED!" << std::endl;
        // }
        // cv::namedWindow("ORB-SLAM2: YOLOv5Detector");
        // cv::imshow("ORB-SLAM2: YOLOv5Detector", showImage);
        // if (cv::waitKey(1) == 27)
        //     break;

        mpFrameDrawer->UploadResults(this);


        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;

    }
    SetFinish();

}


std::vector<torch::Tensor> YOLOv5Detector::non_max_suppression(torch::Tensor preds,
    float score_thresh, float iou_thresh)
{
        std::vector<torch::Tensor> output;
        for (size_t i=0; i < preds.sizes()[0]; ++i)
        {
            torch::Tensor pred = preds.select(0, i);

            // Filter by scores
            torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
            pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
            if (pred.sizes()[0] == 0) continue;

            // (center_x, center_y, w, h) to (left, top, right, bottom)
            pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
            pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
            pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
            pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

            // Computing scores and classes
            std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
            pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
            pred.select(1, 5) = std::get<1>(max_tuple);

            torch::Tensor  dets = pred.slice(1, 0, 6);

            torch::Tensor keep = torch::empty({dets.sizes()[0]});
            torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
            std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
            torch::Tensor v = std::get<0>(indexes_tuple);
            torch::Tensor indexes = std::get<1>(indexes_tuple);
            int count = 0;
            while (indexes.sizes()[0] > 0)
            {
                keep[count] = (indexes[0].item().toInt());
                count += 1;

                // Computing overlaps
                torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
                torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
                for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
                {
                    lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                    tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                    rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                    bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                    widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                    heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
                }
                torch::Tensor overlaps = widths * heights;

                // FIlter by IOUs
                torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
                indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
            }
            keep = keep.toType(torch::kInt64);
            output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
        }
        return output;
}

void YOLOv5Detector::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool YOLOv5Detector::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void YOLOv5Detector::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool YOLOv5Detector::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void YOLOv5Detector::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool YOLOv5Detector::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool YOLOv5Detector::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void YOLOv5Detector::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}



}
