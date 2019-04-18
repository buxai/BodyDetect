#include "BodyType.h"
#include "BodyDetect.h"
#include "ImageSegmentation.h"



bool sortCountersArea(vector<cv::Point> A, vector<cv::Point> B)
{
	return (contourArea(A) > contourArea(B));
}

bool sortX(cv::Point2f A, cv::Point2f B)
{
	return (A.x < B.x);
}

std::vector<cv::Point2f> skeletonBranchPoints(const cv::Mat &thinSrc, unsigned int raudis, unsigned int thresholdMax, unsigned int thresholdMin)
{
	assert(thinSrc.type() == CV_8UC1);
	cv::Mat dst;
	thinSrc.copyTo(dst);
	filterOver(dst);
	int width = dst.cols;
	int height = dst.rows;
	cv::Mat tmp;
	dst.copyTo(tmp);
	std::vector<cv::Point2f> branchpoints;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (*(tmp.data + tmp.step * i + j) == 0)
			{
				continue;
			}
			int count = 0;
			for (int k = i - raudis; k < i + raudis + 1; k++)
			{
				for (int l = j - raudis; l < j + raudis + 1; l++)
				{
					if (k < 0 || l < 0 || k>height - 1 || l>width - 1)
					{
						continue;
					}
					if (*(tmp.data + tmp.step * k + l) == 1)
					{
						count++;
					}
				}
			}

			if (count > thresholdMax)
			{
				cv::Point2f point(j, i);
				branchpoints.push_back(point);
			}
		}
	}
	return branchpoints;
}



vector<cv::Point2f> skeletonEndPoints(cv::Mat &src)
{
	cv::Mat dst;

	vector<cv::Point2f> endpoints;
	cv::Mat k(3, 3, CV_8UC1);

	k.at<uchar>(0, 0) = 1;
	k.at<uchar>(1, 0) = 1;
	k.at<uchar>(2, 0) = 1;
	k.at<uchar>(0, 1) = 1;
	k.at<uchar>(1, 1) = 10;
	k.at<uchar>(2, 1) = 1;
	k.at<uchar>(0, 2) = 1;
	k.at<uchar>(1, 2) = 1;
	k.at<uchar>(2, 2) = 1;

	dst = src.clone();

	filter2D(dst, dst, CV_8UC1, k);
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 1; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) == 11)
			{
				endpoints.push_back(cv::Point2f(j, i));
			}
		}
	}
	return endpoints;
}


vector<cv::Point2f> deleteTooNearPoints(vector<cv::Point2f> src)
{
	vector<cv::Point2f> dst;

	if (!src.empty())
	{
		dst.push_back(src[0]);

		for (int i = 1; i < src.size(); i++)
		{
			int count = 0;
			for (int j = 0; j < j - 1; j++)
			{
				if (sqrt(pow((src[i].x - src[j].x), 2) + pow((src[i].y - src[j].y), 2)) < 5)
				{
					count++;
				}
			}
			if (count < 1)
			{
				dst.push_back(src[i]);
			}
		}
	}
	return dst;
}


vector<cv::Point2f> calcBodyWide(cv::Mat &bodyThreshold, cv::Point2f Center)
{
	vector<cv::Point2f> widePoint;
	for (int i = (Center.x) - 1; i >= 0; i--)
	{
		if (bodyThreshold.at<uchar>(Center.y, i) == 0)
		{
			widePoint.emplace_back(i,Center.y);
			break;
		}
	}

	if (widePoint.empty())
	{
		widePoint.emplace_back(0,Center.y);
	}

	for (int i = (Center.x) + 1; i < bodyThreshold.size().width; i++)
	{
		if (bodyThreshold.at<uchar>(Center.y, i) == 0)
		{
			widePoint.emplace_back(i,Center.y);
			break;
		}
	}

	if (widePoint.size() < 2)
	{
		widePoint.emplace_back(bodyThreshold.size().width - 1, Center.y);
	}
	return widePoint;
}





skeleton FromEdgePoints(vector<cv::Point2f> &skeletonEndPoints, vector<cv::Point2f> &skeletonBranchPoints, cv::Point2f Center, cv::Mat &bodyThreshold)
{

	skeleton skeletonData;
	skeletonData._heart = Center;

	vector<cv::Point2f> test1 = skeletonEndPoints;

	skeletonEndPoints = deleteTooNearPoints(skeletonEndPoints);


	vector<cv::Point2f> test2 = skeletonEndPoints;

	skeletonBranchPoints = deleteTooNearPoints(skeletonBranchPoints);



	cv::Point lowestPoint(0, 0);
	int distance = 0;
	//������˵���͵㣨�����ֱ۾���ĳ��ȣ��������������û��ʵ�����壩
	for (auto& skeletonEndPoint : skeletonEndPoints)
	{
		if (skeletonEndPoint.y > distance)
		{
			lowestPoint = skeletonEndPoint;
			distance = lowestPoint.y;
		}
	}


	//�жϽ���������
	for (auto& skeletonBranchPoint : skeletonBranchPoints)
	{
		//�������ز��ڵ�
		if (skeletonBranchPoint.y < Center.y)
		{
			if (abs(skeletonData.bodyPoint[BodyData_chest].x - Center.x) > abs(skeletonBranchPoint.x - Center.x))	//�ж����п��ܵ��ز��ڵ�
			{
				skeletonData.bodyPoint[BodyData_chest] = skeletonBranchPoint;
			}
		}
		//�����Ǹ����ڵ�
		else
		{
			//�������ڵ�̫����������
			if (skeletonBranchPoint.y > Center.y + abs(Center.y - lowestPoint.y) * 3.0 / 5.0)
			{
				skeletonData.bodyPoint[BodyData_hip] = cv::Point(0, 0);
				continue;
			}
			if (sqrt(pow(skeletonData.bodyPoint[BodyData_hip].x - Center.x,2)+ pow(skeletonData.bodyPoint[BodyData_hip].y - (Center.y*1.3), 2)) > sqrt(pow(
				skeletonBranchPoint.x - Center.x,2)+ pow(skeletonBranchPoint.y - (Center.y*1.3), 2)))	//�ж����п��ܵĸ����ڵ�
			{
				skeletonData.bodyPoint[BodyData_hip] = skeletonBranchPoint;
			}
		}
	}

	vector<cv::Point2f> bodyWide = calcBodyWide(bodyThreshold, Center);



	//�ж�ͷ��λ��
	if (skeletonData.bodyPoint[BodyData_chest] != cv::Point2f(0, 0))
	{
		//���ؿ���Ϊ�жϱ�׼
		for (auto& skeletonEndPoint : skeletonEndPoints)
		{
			if (skeletonEndPoint.y > skeletonData.bodyPoint[BodyData_chest].y)
			{
				//�������ȷ�Χ֮�⣬����
				continue;
			}
			//�ҳ�����������ĵ�
			if (abs(skeletonEndPoint.x - skeletonData.bodyPoint[BodyData_chest].x) < abs(skeletonData.bodyPoint[BodyData_head].x - skeletonData.bodyPoint[BodyData_chest].x))
			{
				if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
				{
					if (sqrt(pow(skeletonEndPoint.x - skeletonData.bodyPoint[BodyData_chest].x,2)+ pow(
						skeletonEndPoint.y - skeletonData.bodyPoint[BodyData_chest].y, 2)) < sqrt(pow(skeletonData.bodyPoint[BodyData_head].x - skeletonData.bodyPoint[BodyData_chest].x, 2) + pow(skeletonData.bodyPoint[BodyData_head].y - skeletonData.bodyPoint[BodyData_chest].y, 2)))
					{
						skeletonData.bodyPoint[BodyData_head] = skeletonEndPoint;
					}
				}
				else
				{
					skeletonData.bodyPoint[BodyData_head] = skeletonEndPoint;
				}

			}
		}
	}
	//���������ĵ���Ϊ�жϱ�׼�����ܲ�׼��
	else
	{
		//���ؿ���Ϊ�жϱ�׼
		for (auto& skeletonEndPoint : skeletonEndPoints)
		{
			if (skeletonEndPoint.y > Center.y)
			{
				//�������ȷ�Χ֮�⣬����
				continue;
			}
			//�ҳ�����������ĵ�
			if (abs(skeletonEndPoint.x - Center.x) < abs(skeletonData.bodyPoint[BodyData_head].x - Center.x))
			{
				if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
				{
					if (abs(skeletonEndPoint.y - Center.y) < abs(skeletonData.bodyPoint[BodyData_head].y - Center.y))
					{
						skeletonData.bodyPoint[BodyData_head] = skeletonEndPoint;
					}
				}
				else
				{
					skeletonData.bodyPoint[BodyData_head] = skeletonEndPoint;
				}
			}
		}
	}

	//ȥ������Եĵ�
	vector<cv::Point2f>::iterator it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_head]);
	if (it != skeletonEndPoints.end())
		skeletonEndPoints.erase(it);


	//ȥ�����ܴ���ĵ�
	if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
	{
		for (it = skeletonEndPoints.begin(); it != skeletonEndPoints.end();)
		{
			if (it->x > bodyWide[0].x && it->x < bodyWide[1].x && it->y < Center.y && it->y > skeletonData.bodyPoint[BodyData_head].y)
			{
				it = skeletonEndPoints.erase(it);
			}
			else
			{
				it++;
			}
		}
	}



	//�ж��ֽţ����������У�
	sort(skeletonEndPoints.begin(), skeletonEndPoints.end(), sortX);

	for (auto& skeletonEndPoint : skeletonEndPoints)
	{
		if (skeletonEndPoint.y >(lowestPoint.y + Center.y) / 2.0)
			continue;
		if (skeletonEndPoint.x < Center.x)
		{
			skeletonData.bodyPoint[BodyData_leftHand] = skeletonEndPoint;
			break;
		}
	}

	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_leftHand]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	for (int i = skeletonEndPoints.size() - 1; i >= 0; i--)
	{
		if (skeletonEndPoints[i].y > (lowestPoint.y + Center.y) / 2.0)
			continue;
		if (skeletonEndPoints[i].x > Center.x)
		{
			skeletonData.bodyPoint[BodyData_rightHand] = skeletonEndPoints[i];
			break;
		}
	}

	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_rightHand]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	for (auto& skeletonEndPoint : skeletonEndPoints)
	{
		if (skeletonEndPoint.y < (bodyThreshold.size().height + Center.y) / 2.0)
			continue;
		skeletonData.bodyPoint[BodyData_leftFoot] = skeletonEndPoint;
		break;
	}


	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_leftFoot]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	for (int i = skeletonEndPoints.size() - 1; i >= 0; i--)
	{
		if (skeletonEndPoints[i].y < (bodyThreshold.size().height + Center.y) / 2.0)
			continue;
		skeletonData.bodyPoint[BodyData_rightFoot] = skeletonEndPoints[i];
		break;
	}

	it = find(skeletonEndPoints.begin(), skeletonEndPoints.end(), skeletonData.bodyPoint[BodyData_rightFoot]);
	if (it != skeletonEndPoints.end())
	{
		skeletonEndPoints.erase(it);
	}


	//�����ز��ƶ�ֵ

	if (skeletonData.bodyPoint[BodyData_head] != cv::Point2f(0, 0))
	{
		skeletonData.bodyPoint[BodyData_chest] = cv::Point2f((Center.x + skeletonData.bodyPoint[BodyData_head].x) / 2, (Center.y + skeletonData.bodyPoint[BodyData_head].y) / 2);
	}

	return skeletonData;
}


