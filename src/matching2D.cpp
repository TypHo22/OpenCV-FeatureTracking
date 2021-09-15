#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, DESCRIPTORFAMILY descriptorFamily, MATCHER matcherType, SELECTOR selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;


    switch (matcherType)
    {
        case(MATCHER::MAT_BF): //brute force matcher
        {
            int normType;
            switch (descriptorFamily)
            {
                case(DESCRIPTORFAMILY::BIN):{normType = cv::NORM_HAMMING; break;}
                case(DESCRIPTORFAMILY::HOG):{normType = cv::NORM_L1; break;}
            }
            matcher = cv::BFMatcher::create(normType, crossCheck);
            break;
        }
        case(MATCHER::MAT_FLANN):
        {
            if (descSource.type() !=CV_32F)
                descSource.convertTo(descSource, CV_32F);

            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            break;
        }
    }

    // perform matching task
    switch (selectorType)
    {
        case(SELECTOR::SEL_NN):
        {
            matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
            break;
        }
        case(SELECTOR::SEL_KNN):
        {
            // k nearest neighbors (k=2)
            vector<vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(descSource, descRef, knn_matches, 2);
            //-- Filter matches using the Lowe's ratio test
            double minDescDistRatio = 0.8;
            for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
            {
                if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                    matches.push_back((*it)[0]);
                }
            }
            break;
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DESCRIPTOR descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    switch (descriptorType)
    {
        case(DESCRIPTOR::BRISK):
        {
            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
            extractor = cv::BRISK::create(threshold, octaves, patternScale);
            break;
        }
        case(DESCRIPTOR::BRIEF):
        {
            int bytes = 32; // Legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
            bool use_orientation = false;// Sample patterns using keypoints orientation, disabled by default.
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
            break;
        }
        case(DESCRIPTOR::ORB):
        {
            int nfeatures = 500;// The maximum number of features to retain.
            float scaleFactor = 1.2f;// Pyramid decimation ratio, greater than 1.
            int nlevels = 8;// The number of pyramid levels.
            int edgeThreshold = 31;// This is size of the border where the features are not detected.
            int firstLevel = 0;// The level of pyramid to put source image to.
            int WTA_K = 2;// The number of points that produce each element of the oriented BRIEF descriptor.
            auto scoreType = cv::ORB::HARRIS_SCORE;// The default HARRIS_SCORE means that Harris algorithm is used to rank features.
            int patchSize = 31;// Size of the patch used by the oriented BRIEF descriptor.
            int fastThreshold = 20;// The fast threshold.
            extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType,
                                        patchSize, fastThreshold);
            break;
        }
        case(DESCRIPTOR::FREAK):
        {
            bool orientationNormalized = true;// Enable orientation normalization.
            bool scaleNormalized = true;// Enable scale normalization.
            float patternScale = 22.0f;// Scaling of the description pattern.
            int nOctaves = 4;// Number of octaves covered by the detected keypoints.
            const std::vector<int> &selectedPairs = std::vector<int>(); // (Optional) user defined selected pairs indexes,
            extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves,
                                                       selectedPairs);
            break;
        }
        case(DESCRIPTOR::AKAZE):
        {
            auto descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;// Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
            int descriptor_size = 0;// Size of the descriptor in bits. 0 -> Full size
            int descriptor_channels = 3;// Number of channels in the descriptor (1, 2, 3)
            float threshold = 0.001f;// Detector response threshold to accept point
            int nOctaves = 4;// Maximum octave evolution of the image
            int nOctaveLayers = 4;// Default number of sublevels per scale level
            auto diffusivity = cv::KAZE::DIFF_PM_G2;// Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER

            extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves,
                                          nOctaveLayers, diffusivity);
            break;
        }
        case(DESCRIPTOR::SIFT):
        {
            int nfeatures = 0;// The number of best features to retain.
            int nOctaveLayers = 3;// The number of layers in each octave. 3 is the value used in D. Lowe paper.
            double contrastThreshold = 0.04;// The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
            double edgeThreshold = 10;// The threshold used to filter out edge-like features.
            double sigma = 1.6;// The sigma of the Gaussian applied to the input image at the octave \#0.
            extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
            break;
        }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << getDescriptorName(descriptorType) << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DETECTOR detectorType, bool bVis)
{
    // select appropriate descriptor
    cv::Ptr<cv::FeatureDetector> detector;

    switch (detectorType)
    {
        case (DETECTOR::AKAZE):
        {
            detector = cv::AKAZE::create();
            break;
        }
        case (DETECTOR::BRISK):
        {
            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
            detector = cv::BRISK::create(threshold, octaves, patternScale);
            break;
        }
        case (DETECTOR::FAST):
        {
            int threshold = 30;// Difference between intensity of the central pixel and pixels of a circle around this pixel
            bool nonmaxSuppression = true;// perform non-maxima suppression on keypoints
            cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;// TYPE_9_16, TYPE_7_12, TYPE_5_8
            detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
            break;
        }
        case (DETECTOR::HARRIS):
        {
            detKeypointsHarris(keypoints,img,bVis);
            break;
        }
        case (DETECTOR::ORB):
        {
            int nfeatures=500;
            float scaleFactor=1.2f;
            int nlevels=8;
            int edgeThreshold=31;
            int firstLevel=0;
            int WTA_K=2;
            cv::ORB::ScoreType scoreType=cv::ORB::HARRIS_SCORE;
            int patchSize=31;
            int fastThreshold=20;
            detector = cv::ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
            break;
        }
        case (DETECTOR::SHITOMASI):
        {
            detKeypointsShiTomasi( keypoints,img,bVis);
            break;
        }
        case (DETECTOR::SIFT):
        {
            int nfeatures = 0;
            int nOctaveLayers = 3;
            double contrastThreshold = 0.04;
            double edgeThreshold = 10;
            double sigma = 1.;
            detector = cv::SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);
            break;
        }
    }
    // perform feature description
    detector->detect(img, keypoints);
    cout << "Detection with n=" << keypoints.size() << endl;
    if (bVis)
    {
        // Visualize the keypoints
        string windowName = getDetectorName(detectorType) + " keypoint detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    if (bVis) {
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }

    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int) dst_norm.at<float>(j, i);
            if (response > minResponse) { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap) {
                        bOverlap = true;
                        if (newKeyPoint.response >
                            (*it).response) {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap) {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
    cout << "Detection with n=" << keypoints.size() << endl;
    // visualize keypoints
    if (bVis) {
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

