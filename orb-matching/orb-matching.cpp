#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat img_left = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_right = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // 检查两张图像是否为空
    assert (img_left.data != nullptr && img_right.data != nullptr);

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1 , descriptors_2;

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 检测Fast 角点的位置
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector ->detect(img_left, keypoints_1);
    detector ->detect(img_right, keypoints_2);

    // 根据角点的位置计算brief描述子
    descriptor ->compute(img_left, keypoints_1, descriptors_1);
    descriptor ->compute(img_right, keypoints_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(img_left, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB_Features", outimg1);
    waitKey(0);

    // STEP 3 根据BRIEF描述子进行匹配，使用汉明距离
    vector<DMatch> matches;
    matcher -> match(descriptors_1, descriptors_2, matches);

    // STEP 4 筛选匹配的特征点
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // visualize
    Mat img_match;
    Mat img_goodmatch;

    drawMatches(img_left, keypoints_1, img_right, keypoints_2, matches, img_match);
    drawMatches(img_left, keypoints_1, img_right, keypoints_2, good_matches, img_goodmatch);
    imshow("good matches", img_goodmatch);
    waitKey(0);
    return 0;
}