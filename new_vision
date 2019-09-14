//
//  surf_to_pic.cpp
//  feature_extraction
//
//  Created by xiaoyu on 2019/9/5.
//  Copyright © 2019 xiaoyu. All rights reserved.
//
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using std::cout;
using std::endl;

const char* params =
"{ help h |                          | Print help message. }"
"{ input1 | /Users/Tribesman/Desktop/Xcode/20190726--read_video_subtract_background/test2.mov | Path to input video 1. }"
"{ input2 | /Users/Tribesman/Desktop/Xcode/20190912--simple_surf_to_video/VID_20181225_172152.mp4 | Path to input video 2. }";
int main( int argc, char* argv[] )
{
    CommandLineParser parser(argc, argv, params);
    //不用IplImage*， 用Mat，IplImage*是之前版本的图像的存储
    //读取视频文件
    VideoCapture capture(parser.get<String>("input1"));
    
    Mat frameA, frameB, frameC;
    capture >> frameA;
    
    while (true) {
        //把后面的帧都赋予frameB
        capture >> frameB;
        
        //得到当前一帧的号码
        stringstream ss;
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        //获得视频的帧数
        int framenum = capture.get(CAP_PROP_FRAME_COUNT);
        cout << framenum << endl;
        
        //读取照片，并且灰度化，
        //转化为灰度图像，第一帧转换为frameC，后面的帧都转换为frameB，然后所有后面的帧都和第一帧frameC进行对比。6的意思是rgb_to_gray
        cvtColor(frameA,frameC,6);
        cvtColor(frameB,frameB,6);
        //如果没有读入帧数，则输出错误提示
        if ( frameC.empty() || frameB.empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            parser.printMessage();
            return -1;
        }
        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        // 之前设定的是400
        int minHessian = 400;
        Ptr<SURF> detector = SURF::create( minHessian );
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute( frameC, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( frameB, noArray(), keypoints2, descriptors2 );
        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        //-- Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        //这个是在原来的视频上面，显示当前帧的号码
        //先画一个底色的框框
        rectangle(frameB, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(0,0,0), -1);
        //然后在上面显示当前帧的数字
        putText(frameB, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(255,255,255));
        
        //-- Draw matches
        Mat img_matches;
        drawMatches( frameC, keypoints1, frameB, keypoints2, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
        //print一下当前帧的号码
//        cout << frameNumberString << endl;
        
        //想办法把坐标的平均值输出出来
        //给特征点分配空间
        int ptCount = (int)knn_matches.size();
        Mat p1(ptCount, 2, CV_32F);
        Mat p2(ptCount, 2, CV_32F);
        // 把Keypoint转换为Mat
        //现在queryIdx里面有点儿问题，是DMatch里面的内容
        Point2f pt1,pt2;
        //c++里面矩阵的概念相对应的，应该是数组
        double framex[ptCount], framey[ptCount];
        //每一帧里面，x, y位移的和
        double sum_oneframe_x = 0, sum_oneframe_y = 0;
        //每一帧里面，x, y位移的平均值
        double ave_x = 0, ave_y = 0;
        //在每一帧里面循环，把一帧里面的所有的特征点都提取出来
        for (int i = 0; i<ptCount; i++)
        {
            //原图特征点的坐标
            pt1 = keypoints1[knn_matches[i][0].queryIdx].pt;
            //train图中的特征点的坐标，也就是后续帧的坐标
            pt2 = keypoints2[knn_matches[i][1].trainIdx].pt;
            //train相对于原图的位移
            framex[i] = pt2.x - pt1.x;
            framey[i] = pt2.y - pt1.y;
            //输出看看对不对，感觉是对着的
//            cout << framex[i] << '\t' << framey[i] << endl;
            //把这个位移累加了，跳出循环的时候/pptCount，得到平均值
            sum_oneframe_x = sum_oneframe_x + framex[i];
            sum_oneframe_y = sum_oneframe_y + framey[i];
        }
        //平均值
        ave_x = sum_oneframe_x / ptCount;
        ave_y = sum_oneframe_y / ptCount;
        //看看对不对
        cout << ave_x << '\t' << ave_y << endl;
        ofstream oFile("/Users/Tribesman/Desktop/Xcode/20190912--simple_surf_to_video/feature_extraction/try1.txt", ios_base::app);
        //测试是否打开了该文件
        if (! oFile){//打开失败
            cerr << "cannot open copy.out for output\n";
            exit(-1);
        }
        oFile << frameNumberString << '\t' <<ave_x << '\t' << ave_y << std::endl;//每次写完一个矩阵以后换行
    }
    
#else
    int main()
    {
        std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
        return 0;
    }
    return 0;
}

#endif
}
