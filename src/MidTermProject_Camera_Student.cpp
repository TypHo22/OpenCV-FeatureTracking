/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */



int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time


    //Performance
    std::vector<int> keyPointsAmount;
    std::vector<float> keyPointAvgSize;
    std::vector<long> durationKeyPointDetection;
    std::vector<long> durationDescriptorExtraction;
    std::vector<int>  keyPointMatches;

    /*
        _____ ______ _______ _______ _____ _   _  _____  _____
       / ____|  ____|__   __|__   __|_   _| \ | |/ ____|/ ____|
      | (___ | |__     | |     | |    | | |  \| | |  __| (___
       \___ \|  __|    | |     | |    | | | . ` | | |_ |\___ \
       ____) | |____   | |     | |   _| |_| |\  | |__| |____) |
      |_____/|______|  |_|     |_|  |_____|_| \_|\_____|_____/
    */
        DETECTOR detectorType(DETECTOR::SIFT);
        DESCRIPTOR descriptorType = DESCRIPTOR::BRISK;
        DESCRIPTORFAMILY descriptorFamily = DESCRIPTORFAMILY::BIN;//SIFT DESCRIPTOR comes from HOG Family
        bool bVis = false;            // visualize results

        MATCHER matcherType = MATCHER::MAT_BF;        // MAT_BF, MAT_FLANN

        SELECTOR selectorType = SELECTOR::SEL_KNN;       // SEL_NN, SEL_KNN
        bool logPerformance = true;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;

        if (dataBuffer.size() > dataBufferSize) {
            dataBuffer.pop_front();
        }
        dataBuffer.push_back(frame);



        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = "SHITOMASI";


        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        /**
          @author
          I ignored string-based detection
        */
        auto detection_start = std::chrono::high_resolution_clock::now();
        switch (detectorType)
        {
            case (DETECTOR::HARRIS):{detKeypointsHarris(keypoints, imgGray, bVis);break;}
            case (DETECTOR::SHITOMASI):{detKeypointsShiTomasi(keypoints, imgGray, bVis);break;}
            default: {detKeypointsModern(keypoints, imgGray, detectorType, bVis);break;}//BRISK, SIFT, AKAZE, ORB, FAST
        }
        auto detection_end = std::chrono::high_resolution_clock::now();
        durationKeyPointDetection.push_back((detection_end - detection_start)/std::chrono::microseconds(1));


        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            for(auto p = keypoints.begin(); p != keypoints.end(); ++p)
            {
                if(!vehicleRect.contains(p->pt))
                {
                    keypoints.erase(p);
                    p--;
                    continue;
                }
            }
        }
        keyPointsAmount.push_back(keypoints.size());
        float avgSize = 0;
        for(auto& p : keypoints)
        {
            avgSize += p.size;
        }
        keyPointAvgSize.push_back(avgSize / keypoints.size());


        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            switch (detectorType)
            {
                case (DETECTOR::SHITOMASI):{keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());;break;}
                default:{break;}
            }

            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        //string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        auto descriptors_start = std::chrono::high_resolution_clock::now();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        auto descriptors_end = std::chrono::high_resolution_clock::now();
        durationDescriptorExtraction.push_back((descriptors_end - descriptors_start)/std::chrono::microseconds(1));
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorFamily, matcherType, selectorType);
            keyPointMatches.push_back(matches.size());

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    if(logPerformance)
    {
        std::cout<<"========================"<<std::endl;
        std::cout<<"========================"<<std::endl;
        std::cout<<"Used detector: " << getDetectorName(detectorType) << std::endl;
        std::cout<<"Used descriptor: " << getDescriptorName(descriptorType) << std::endl;

        std::cout<<"<-keyPointsFound->"<<std::endl;
        int keyPointAmount = 0;
        for_each(keyPointsAmount.begin(),keyPointsAmount.end(),[&keyPointAmount](const int amount)
        {
            keyPointAmount += amount;
        });
        std::cout<<" keypoints detected: \n"<< keyPointAmount <<std::endl;

        std::cout<<"<-keyPointsMatches->"<<std::endl;
        int keyPointMatched = 0;
        for_each(keyPointMatches.begin(),keyPointMatches.end(),[&keyPointMatched](const int amount)
        {
            keyPointMatched += amount;
        });
        std::cout<<"keypoints matched: \n"<< keyPointMatched <<std::endl;

        std::cout<<"<-KeyPointsSize->"<<std::endl;
        float avgKeyPointSize;
        for_each(keyPointAvgSize.begin(),keyPointAvgSize.end(),[&avgKeyPointSize](const float size)
        {
            avgKeyPointSize += size;
        });
        std::cout<<"Average keypointssize detected: \n"<< avgKeyPointSize / keyPointAvgSize.size() <<std::endl;

        std::cout<<"<-DurationKeyPointDetection->"<<std::endl;
        long avgDetection = 0;
        for_each(durationKeyPointDetection.begin(),durationKeyPointDetection.end(),[&avgDetection](const float time)
        {
            avgDetection += time;
        });
        std::cout<<"Average keypoint detection time: \n"<< avgDetection / durationKeyPointDetection.size() <<" µs"<<std::endl;

        std::cout<<"<-Duration descriptor extraction->"<<std::endl;
        long avgExtraction = 0;
        for_each(durationDescriptorExtraction.begin(),durationDescriptorExtraction.end(),[&avgExtraction](const float time)
        {
            avgExtraction += time;
        });
        std::cout<<"Average Descriptor extraction time: \n"<< avgExtraction / durationDescriptorExtraction.size() <<" µs"<<std::endl;





        std::cout<<"========================"<<std::endl;
        std::cout<<"========================"<<std::endl;
    }

    return 0;
}
