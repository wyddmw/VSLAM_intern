#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <eigen3/Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

// declare some functions
void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);

void find_feature_matches(
    const Mat &img_left, const Mat &img_right,
    vector<KeyPoint> &keypoints_1,
    vector<KeyPoint> &keypoints_2,
    vector<DMatch> &matches
);

// 像素坐标转换为归一化坐标
cv::Point2d pixel2cam(const Point2d &p, const Mat &K); 

int main(int argc, char** argv)
{
    // double fx = 518.0, fy = 519.0, cx = 325.5, cy = 253.5;
    Mat K = (Mat_<double>(3, 3) << 518.0, 0, 325.5, 0, 519.0, 253.5, 0, 0, 1);
    // Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    double depthScale = 10000.0;

    // 读取图像序列
    vector<cv::Mat> colorImgs, depthImgs;
    // vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;   // 相机位姿
    bool pose_flag = false;

    for (int i = 0; i <4; i++)
    {
        boost::format fmt("./%s/%s.%s");        // 图像文件格式
        colorImgs.push_back(cv::imread((fmt%"color"%(i+2)%"png").str(), CV_LOAD_IMAGE_COLOR));
        depthImgs.push_back(cv::imread((fmt%"depth"%(i+2)%"pgm").str(), -1));
    }

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // 全局的点云图
    PointCloud::Ptr pointCloud(new PointCloud);

    // 初始化旋转向量 各个轴上的旋转等于0
    Matrix3d R = AngleAxisd(0, Vector3d(0, 0, 0)).toRotationMatrix();
    Sophus::SO3d SO3_R(R);
    cout << SO3_R.matrix() << endl;
    Eigen::Vector3d t(0, 0, 0);
    Sophus::SE3d pose_gn(R, t);
    cout << pose_gn.matrix() << endl;
    Sophus::SE3d pose_last = pose_gn;

    for (int i=0; i<4; i++)
    {
        PointCloud::Ptr current(new PointCloud);

        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        
        if (i != 0)
        {
            // 位姿估计
            vector<KeyPoint> keypoints_1, keypoints_2;
            vector<DMatch> matches;
            cv::Mat color_last = colorImgs[i-1];
            cv::Mat depth_last = depthImgs[i-1];
            find_feature_matches(color_last, color, keypoints_1, keypoints_2, matches);
            cout << "find " << matches.size() << " matches " << endl;

            // 建立3D坐标
            vector<Point3f> pts_3d;
            vector<Point2f> pts_2d;

            for (DMatch m:matches)
            {
                ushort d = depth.ptr<unsigned short>(int (keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];        // 获得对应的深度信息
                if (d == 0)
                {
                    continue;
                }
                float dd = d / depthScale;
                Point2d p1 = pixel2cam(keypoints_2[m.trainIdx].pt, K);          // 前一帧的图像转换到下一帧的图像上
                pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
                pts_2d.push_back(keypoints_1[m.queryIdx].pt);
            }

            cout << "3d-2d pairs: " << pts_3d.size() << endl;           

            VecVector3d pts_3d_eigen;
            VecVector2d pts_2d_eigen;

            for (size_t i = 0; i < pts_3d.size(); ++i)
            {
                pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
                pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
            }

            bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
            for(int v=0; v<color.rows; v++)
            {
                for (int u=0; u<color.cols; u++)
                {
                    unsigned int d = depth.ptr<unsigned short> (v)[u];
                    if (d == 0)
                    {
                        continue;
                    }
                    Eigen::Vector3d point;
                    point[2] = double(d)/depthScale; 
                    point[0] = (u-cx)*point[2]/fx;
                    point[1] = (v-cy)*point[2]/fy; 
                    Eigen::Vector3d pointWorld;
                    
                    if (pose_flag)
                    {
                        pointWorld = pose_last * pose_gn *  point;
                    }
                    else
                    {
                        pointWorld = pose_gn *  point;
                        pose_last = pose_gn;
                    }
                    
                    PointT p ;
                    p.x = pointWorld[0];
                    p.y = pointWorld[1];
                    p.z = pointWorld[2];
                    p.b = color.data[ v*color.step+u*color.channels() ];
                    p.g = color.data[ v*color.step+u*color.channels()+1 ];
                    p.r = color.data[ v*color.step+u*color.channels()+2 ];
                    current->points.push_back( p );
                }
            }

            pose_flag = true;
            if (i >1)
            {
                pose_last = pose_last * pose_gn;
            }
        }
        
        else
        {
            cv::Mat color = colorImgs[0];
            cv::Mat depth = depthImgs[0];

            for (int v=0; v<colorImgs[0].rows; v++)
            {
                for (int u=0; u<colorImgs[0].cols; u++)
                {
                    unsigned int d = depth.ptr<unsigned short>(v)[u];
                    if (d == 0){
                        continue;
                    }

                    Eigen::Vector3d point;
                    point[2] = double(d) / depthScale;
                    point[0] = (u - cx) * point[2] / fx;
                    point[1] = (v - cy) * point[2] / fy;

                    PointT p;
                    // p.head<3>() = point;
                    p.x = point[0];
                    p.y = point[1];
                    p.z = point[2];
                    p.b = color.data[ v*color.step+u*color.channels() ];
                    p.g = color.data[ v*color.step+u*color.channels()+1 ];
                    p.r = color.data[ v*color.step+u*color.channels()+2 ];
                    current->points.push_back( p );
                    // pointCloud->points.push_back(p);   
                }
            }
        }

        // depth filter and statistical removal
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(10);
        statistical_filter.setStddevMulThresh(1.0);
        statistical_filter.setInputCloud(current);
        statistical_filter.filter(*tmp);
        (*pointCloud) += *tmp;
    }

    pointCloud->is_dense = false;

    // voxel filter
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.01;
    voxel_filter.setLeafSize(resolution, resolution, resolution);
    PointCloud::Ptr tmp(new PointCloud);
    voxel_filter.setInputCloud(pointCloud);
    voxel_filter.filter(*tmp);
    tmp->swap(*pointCloud);

    cout << "after filter " << pointCloud->size() << endl;

    pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
    return 0;    
}

cv::Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        // p.x - cx / fx,
        // p.y - 
    );      // 转换为归一化像素点坐标
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                                                            std::vector<KeyPoint> &keypoints_1,
                                                            std::vector<KeyPoint> &keypoints_2,
                                                            std::vector<DMatch> &matches)
{
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // Detect Oriented Fast location
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // 计算描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // 根据brief 描述子进行匹配的计算
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    // 筛选匹配的点对
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("--Max dist : %f\n", max_dist);
    printf("--Min_dist : %f\n", min_dist);

    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if(match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}


void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose){
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        const int iterations = 10;
        double cost = 0, lastCost = 0;
        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        for (int iter=0; iter < iterations; iter++)
        {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Vector6d b = Vector6d::Zero();

            cost = 0;
            for (int i=0; i < points_3d.size(); i++)
            {                
                Eigen::Vector3d pc = pose * points_3d[i];               // * 进行了运算符的重载，可以实现齐次坐标向非齐次坐标的转换
                // cout << pc << endl << i << endl;
                double inv_z = 1.0 / pc[2];
                double inv_z2 = inv_z * inv_z;
                Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy );            // 投影到2D平面
                // 计算重投影误差
                Eigen::Vector2d e = points_2d[i] - proj;

                cost += e.squaredNorm();
                Eigen::Matrix<double, 2, 6> J;      // matrix矩阵 主要三个参数，分别是数据的类型，行列

                J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;
                
                H += J.transpose() * J;
                b += -J.transpose() * e;
            }

            Vector6d dx;
            dx = H.ldlt().solve(b);

            if (isnan(dx[0])){
                cout << "result is nan!" << endl;
                break;
            }

            if (iter > 0 && cost >= lastCost){
                cout << "cost: " << cost << ", last cost: " << lastCost << endl;
                break;
            }

            pose = Sophus::SE3d::exp(dx) * pose;
            lastCost = cost;

            cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;

            if (dx.norm() < 1e-6){
                break;
            }
        }
        // cout << "pose by gn: \n" << pose.matrix() << endl;
        // cout << pose.matrix().size() << endl;
    }