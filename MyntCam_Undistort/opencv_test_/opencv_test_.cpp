#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "mynteyed/camera.h"
#include "mynteyed/utils.h"

using namespace mynteyed;
using namespace std;
using namespace cv;

int main(int argc, char const* argv[]) {
	Camera cam;
	DeviceInfo dev_info;
	Mat cameramatrix = Mat::zeros(3, 3, CV_64F);
	Mat distcoeffs = Mat::zeros(5, 1, CV_64F);

	// 保存摄像头内容到视频 
	Mat src;
	int fps = 30;
	int frameWidth = 1280;
	int frameHeight = 720;
	// 压缩帧的编解码方式
	VideoWriter writer("outdoor.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameWidth, frameHeight), 1);

	if (!util::select(cam, &dev_info)) {
		return 1;
	}
	util::print_stream_infos(cam, dev_info.index);
	std::cout << "Open device" << dev_info.index << "," << dev_info.name << std::endl << std::endl;

	OpenParams params(dev_info.index);
	{
		params.framerate = 30;
		params.stream_mode = StreamMode::STREAM_1280x720;	
		params.ir_intensity = 0;
	}

	cam.EnableImageInfo(true);
	cam.Open(params);
	if (!cam.IsOpened())
	{
		std::cerr << "Open Camera failed" << std::endl;
		return 1;
	}

	bool in_ok, ex_ok;
	auto hd_intrinsics = cam.GetStreamIntrinsics(StreamMode::STREAM_1280x720);

	cameramatrix.at<double>(0, 0) = hd_intrinsics.left.fx;
	cameramatrix.at<double>(0, 2) = hd_intrinsics.left.cx;
	cameramatrix.at<double>(1, 1) = hd_intrinsics.left.fy;
	cameramatrix.at<double>(1, 2) = hd_intrinsics.left.cy;		// 读取相机的参数
	cameramatrix.at<double>(2, 2) = 1;

	distcoeffs.at<double>(0, 0) = hd_intrinsics.left.coeffs[0];
	distcoeffs.at<double>(1, 0) = hd_intrinsics.left.coeffs[1];
	distcoeffs.at<double>(2, 0) = hd_intrinsics.left.coeffs[2];
	distcoeffs.at<double>(3, 0) = hd_intrinsics.left.coeffs[3];	// 根据相机相关的参数进行图像
	distcoeffs.at<double>(4, 0) = 0;

	// 判断摄像头是否正确打开
	std::cout << "Open device success" << endl;
	bool is_left_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_LEFT_COLOR);
	if (is_left_ok)
	{
		namedWindow("left_distort");
		namedWindow("left_undistort");
	}
	for (;;)
	{
		cam.WaitForStream();
		if (is_left_ok)
		{
			auto left_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
			if (left_color.img)
			{
				Mat left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
				Mat new_left;
				undistort(left, new_left, cameramatrix, distcoeffs);
				imshow("left_distort", left);
				imshow("left_undistort", new_left);
				writer << new_left;
			}
		}
		char key = static_cast<char>(waitKey(1));
		if (key == 'q')
		{
			break;
		}
	}
	cam.Close();
	writer.release();
	return 0;
}