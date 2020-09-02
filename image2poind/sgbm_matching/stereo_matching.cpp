#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Geometry>
#include <boost/format.hpp>
#include <pcl/point_types.h>
// #include <pcl-1.8/pcl/point_types.h>
#include <pcl/io/pcd_io.h>
// #include <pcl-1.8/pcl/pcd_io.h>
// #include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv){
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573;
    cv::Mat left = cv::imread(argv[1]);
    cv::Mat right = cv::imread(argv[2]);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
    cv::Mat disparity_sgbm, disparity;
    sgbm ->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    

    typedef pcl::PointXYZRGB PointT;                            // 单个点云数据的存储形式
    typedef pcl::PointCloud<PointT> PointCloud;     // 存储整个的点云
    
    PointCloud::Ptr pointCloud(new PointCloud);     // 智能指针
    for (int v = 0; v < left.rows; v++)
    {
        for (int u = 0; u < left.cols; u++)
        {
            Vector3d point;
            double d = fx * b / (disparity.at<float>(v, u));        // 因为没有注意到符号的问题，所以导致产生了数值的溢出
            if ( d <= 0 || d >= 96)
            { 
                continue;
            }
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            point[2] =  (d);
            cout << point[2] << endl;
            point[0] = x * point[2];
            point[1] = y * point[2];

            PointT p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
            p.b = left.data[v*left.step + u*left.channels()];
            p.g = left.data[v*left.step + u*left.channels() + 1];
            p.r = left.data[v*left.step + u*left.channels() + 2];
            pointCloud->points.push_back(p);            
        }
    }
    pointCloud ->is_dense = false;
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;

}