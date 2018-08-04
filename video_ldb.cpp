#include<iostream>
#include<iomanip>
#include<string>
#include<vector>
#include<algorithm>  // sort
#include<fstream>
#include<chrono>     // timing

#include "ldb.h"
#include "gms_matcher.h"

#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION > 2
  #include<opencv2/xfeatures2d.hpp>
#endif

using namespace std;
//using namespace cv;

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cerr <<"usage: " << argv[0] <<" video_file , (optional) skipframes"
                  << std::endl;
        return 1;
    }
    int skipframes = 0;
    if(argc == 3)
        sscanf(argv[2],"%d",&skipframes);
    std::cout <<"OpenCV Version: "<< CV_MAJOR_VERSION <<"."<<CV_MINOR_VERSION << std::endl;
    std::string video_file(argv[1]);

    cv::VideoCapture capture(video_file);
    if(!capture.isOpened())
    {
        std::cerr <<"fail open " << video_file << std::endl;
        return 2;
    }

    int W = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int H = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    cv::Size imgSize(W,H);

    std::vector<cv::KeyPoint> vKpt1,vKpt2;
    cv::Mat descriptors_1,descriptors_2;
    cv::Mat prevFrame,currFrame;
    cv::Mat imCurrent_rgb;

    int ni = 0;
    bool setOK = capture.set(CV_CAP_PROP_POS_FRAMES,(double)ni);
    LDB ldb(48);
    bool ldb_flag = 1; // true: compute orientation, false: no orientation

    bool bCrossCheck(false);
    cv::BFMatcher bfMatcher(cv::NORM_HAMMING, bCrossCheck); //brute-force

    while(capture.read(imCurrent_rgb))
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        cv::cvtColor(imCurrent_rgb,currFrame,CV_BGR2GRAY);

        cv::FAST(currFrame,vKpt2,20,true);
        ldb.compute(currFrame,vKpt2,descriptors_2,ldb_flag);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double detect_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        printf("detect time: %f ms\n",detect_time*1000);

        if(!prevFrame.empty())
        {
            //-- Keypoint Matching
            //for each key point, find two mached points: the best point and the better point
            vector< vector<cv::DMatch> > matches;//all matched pairs
            vector<cv::DMatch> good_matches;//good matched pairs

            bfMatcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

            for(unsigned i=0; i<matches.size(); i++)
            {
                //if the best point is ditinguished from the better one, keep it
                if(matches[i][0].distance < 0.8*matches[i][1].distance){
                    good_matches.push_back(matches[i][0]);
                }
            }
            gms_match(imgSize,imgSize,vKpt1,vKpt2,good_matches);
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            double match_time = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
            printf("match time: %f ms \n",match_time);

            //draw the good-macthed pairs
//            cv::Mat out_img;
//            cv::drawMatches(prevFrame, vKpt1, currFrame, vKpt2, good_matches, out_img);
            for(unsigned i=0; i<good_matches.size(); i++)
            {
                cv::line(imCurrent_rgb,
                         vKpt1[good_matches[i].queryIdx].pt,
                         vKpt2[good_matches[i].trainIdx].pt,
                         cv::Scalar(0,255,0));
            }
            cv::namedWindow("ldb match",CV_WINDOW_NORMAL);
            cv::imshow("ldb match", imCurrent_rgb);
            char key = cv::waitKey(30);
            if(key == 'q' )
                break;

            matches.clear();
            good_matches.clear();
        }

        prevFrame = currFrame.clone();
        vKpt1 = vKpt2;
        descriptors_1 = descriptors_2.clone();

        vKpt2.clear();
        descriptors_2.release();

        ni += skipframes;
        setOK = capture.set(CV_CAP_PROP_POS_FRAMES,(double)ni);

        if(!setOK)
            break;
// display
//        cv::drawKeypoints(imCurrent_rgb,vKpt2,imCurrent_rgb,
//                          cv::Scalar::all(-1),
//                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
//        cv::namedWindow("frame",CV_WINDOW_NORMAL);
//        cv::imshow("frame",imCurrent_rgb);
//        if('q' == cv::waitKey(10))
//            break;
    }

    return 0;

}