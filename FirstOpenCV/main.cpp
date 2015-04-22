#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cv.h>


using namespace cv;
using namespace std;

void colorReduceI(Mat &image, int div);
void colorReduceK(Mat &image, int div);
void salt(Mat &, int );
void sharpen(const cv::Mat &image, cv::Mat &result);
void sharpen2D(const cv::Mat &image, cv::Mat &result);

//own
void createChessBoard(Mat &image, int width, int height);


class ColorDetector
{
	public:
		ColorDetector() : minDist(240)
		{
			target[0] = target[1] = target[2];
		}
			// Sets the color to be detected
		void setTargetColor(unsigned char red, unsigned char green, unsigned char blue) {
				// BGR order
				target[2]= red;
				target[1]= green;
				target[0]= blue;
		}

		Mat process(const Mat &image);

	private:
	
		void setColorDistanceThreshold(int distance)
		{
			if(distance<0)
				distance=0;

			minDist = distance;
		}

		int getColorDistanceThreshold() const 
		{
			return minDist;
		}

		int getDistance(const cv::Vec3b& color) const
		{
			return abs(color[0]- target[0]) + abs(color[1]- target[1]) + abs(color[2]- target[2]); 
		}

	
		// Sets the color to be detected
		void setTargetColor(cv::Vec3b color) {
				target= color;
		}
		// Gets the color to be detected
		Vec3b getTargetColor() const {
				return target;
		}

		


		// minimum acceptable distance
		int minDist;
		// target color
		Vec3b target;
		// image containingg resulting binary map
		Mat result;

};



Mat ColorDetector::process(const Mat &image)
{
	result.create(image.rows, image.cols,CV_8U);
	Mat_<Vec3b>::const_iterator it = image.begin<Vec3b>();
	Mat_<Vec3b>::const_iterator itend = image.end<Vec3b>();
	Mat_<uchar>::iterator itout = result.begin<uchar>();

	for(; it!= itend; ++it, ++itout)
	{
		if(getDistance(*it)<minDist)
		{
			*itout = 255;
		}
		else
		{
			*itout = 0;
		}
	}

	return result;
}

void main()
{
	
	ColorDetector cdetect;
	Mat image = imread("E:\\lena.png",CV_LOAD_IMAGE_COLOR);
	
	cdetect.setTargetColor(130,190,230);

	namedWindow("result");
	imshow("result", cdetect.process(image));
	waitKey();

}



















//	Mat image = imread("E:\\lena.png",CV_LOAD_IMAGE_COLOR);
//
//	Mat image2 = imread("E:\\image1.png",CV_LOAD_IMAGE_COLOR);
//	Mat output;// = imread("E:\\lena.png",CV_LOAD_IMAGE_COLOR);
//	output.create(image.size(),image.type());
//	namedWindow("Lena",WINDOW_NORMAL);
//	namedWindow("Lena",CV_WINDOW_AUTOSIZE);

//	
//	cv::Mat greyMat;
////	cv::cvtColor(image, greyMat, CV_BGR2GRAY);
//	//cv::addWeighted(image,0.7,image2,0.9,0.,output);
//	output= 0.7*image+0.9*image2;
//	//sharpen2D(greyMat,output);
//	imshow("Lena",output);
//
//	waitKey();

void createChessBoard(Mat &image, int width, int height)
{
	image = Mat(width,height,CV_8U);

	for(int i=0; i<width; i++)
	{
		uchar* data = image.ptr<uchar>(i);

		for(int j=0; j<height; j++)
		{
			if(i%2==0)
			{
				if(j%2==0)	data[j] = 0;
				else        data[j] = 255;
			}
			else
			{
				if(j%2==0)	data[j] = 255;
				else        data[j] = 0;
			}
		}
	}
}

void colorReduceK(Mat &image, int div=64)
{
	int nl = image.rows; //number of lines
	int nc = image.cols * image.channels(); //number of elements per line

	for(int j=0; j<nl; j++)
	{
		uchar* data = image.ptr<uchar>(j); //get the address of the row

		for(int i=0; i<nc; i++)
		{
			int b = data[i];
			data[i] = data[i]/div * div + div/2;

		}
	}
}

// salt pepper
void salt(Mat &image, int n)
{
	for(int k=0; k<n;k++)
	{
		int i = rand()%image.cols;
		int j = rand()%image.rows;

		if(image.channels()==1)
		{
			image.at<uchar>(j,i) = 255;	// we need to specify to be sure the type to be sure
		}
		else if(image.channels()==3)
		{
			image.at<Vec3b>(j,i)[0] = 255;
			image.at<Vec3b>(j,i)[1] = 255;
			image.at<Vec3b>(j,i)[2] = 255;
		}

	}
}

void sharpen(const cv::Mat &image, cv::Mat &result)
{
	result.create(image.size(),image.type());

	for(int j=1; j<image.rows-1; j++) // for all rows except first and last
	{
		const uchar* previous = image.ptr<const uchar>(j-1);
		const uchar* current = image.ptr<const uchar>(j);
		const uchar* next = image.ptr<const uchar>(j+1);

		uchar *output = result.ptr<uchar>(j);

		for(int i=1; i<image.cols-1; i++)
		{
			*output++ = cv::saturate_cast<uchar>(5*current[i] - current[i-1] - current[i+1] - previous[i] - next[i]);
		}
	}

	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows-1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols-1).setTo(cv::Scalar(0));

}


void sharpen2D(const cv::Mat &image, cv::Mat &result)
{
	cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));

	// assign kernel values
	kernel.at<float>(1,1) = 8.0;
	kernel.at<float>(0,1) = -1.0;
	kernel.at<float>(2,1) = -1.0;
	kernel.at<float>(1,0) = -1.0;
	kernel.at<float>(1,2) = -1.0;

	//filter the image
	cv::filter2D(image,result,image.depth(),kernel);
}

void colorReduceI(Mat &image, int div=64)
{
	Mat_<Vec3b>::iterator it = image.begin<Vec3b>(); //obtain iterator with initial position
	Mat_<Vec3b>::iterator itend = image.end<Vec3b>(); // obtain iterator with last position

	for(; it!=itend; it+=10)
	{
		(*it)[0] = (*it)[0]/div * div + div/2;
		(*it)[1] = (*it)[1]/div * div + div/2;
		(*it)[2] = (*it)[2]/div * div + div/2;
	}


}