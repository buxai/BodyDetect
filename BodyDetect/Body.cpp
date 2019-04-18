#include "Body.h"
#include "BodyType.h"
#include "BodyDetect.h"
#include "ImageSegmentation.h"

bool CJcCalBody::recognizeImage(cv::Mat& img)
{
	//读入已二值化并裁剪过的黑白图

	const DWORD startTime = GetCurrentTime();
	cv::Mat imgCopy;
	img.copyTo(imgCopy);
	vector<vector<cv::Point> > contours;
	findContours(imgCopy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), sortCountersArea);

	for (auto it = contours.begin(); it != contours.end();)
	{
		if (contourArea(*it) < 1000)
		{
			it = contours.erase(it);
		}
		else
		{
			++it;
		}
	}

	if (!contours.empty())
	{

		vector<skeleton> PersonSkeleton;	//每个人对应的骨架
		for (int i = 0; i < contours.size(); i++)
		{
			vector<cv::Mat> PersonMat;	//每个人对应的Mat
			cv::Rect PersonRect = boundingRect(contours[i]);

			cv::Mat cutFrameRectCopy;
			img(PersonRect).copyTo(cutFrameRectCopy);


			resize(cutFrameRectCopy, cutFrameRectCopy, cv::Size(150.0*PersonRect.width / PersonRect.height, 150.0));

			int area = contourArea(contours[i]) * pow((150.0 / PersonRect.height), 2);
			int personNumber = min(area / 4000.0, cutFrameRectCopy.size().width / 52.5);



			if (personNumber >= 2)
			{

				//表示当前Rect人数可能多于1人，需要裁剪
				int PersonWidth = cutFrameRectCopy.size().width / personNumber;	//得到每个人的大概宽度


				vector<cv::Point> heightPoint;	//每个人的重心
				for (int j = 0; j < personNumber; j++)
				{
					//处理每幅图
					cv::Rect cutPersonRect(j*PersonWidth, 0, PersonWidth, cutFrameRectCopy.size().height);
					cv::Mat PersonTemp;
					cutFrameRectCopy(cutPersonRect).copyTo(PersonTemp);	//存储每个人的临时图像

																		//算出每个图片的重心
					vector<vector<cv::Point>> PersonTempContours;
					findContours(PersonTemp, PersonTempContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
					std::sort(PersonTempContours.begin(), PersonTempContours.end(), sortCountersArea);
					cv::Moments mu = moments(PersonTempContours[0]);
					if (mu.m00 == 0)
					{
						continue;
					}
					else
					{
						//存储每张图片的重心点
						cv::Point mc = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
						heightPoint.emplace_back(j*PersonWidth + mc.x, mc.y);
						//circle(cutFrameRectCopy, Point(j*PersonWidth + mc.x, mc.y), 2, Scalar(0, 0, 0));
					}
				}



				vector<int> cutEdge;
				cutEdge.push_back(0);

				//寻找两张图片之间需要切割的地方
				for (int j = 0; j < heightPoint.size() - 1; j++)
				{

					int minWhiteLength = 9999;
					int cutX;
					for (int k = heightPoint[j].x; k < heightPoint[j + 1].x; k++)
					{
						int countWhiteLength = 0;
						for (int l = 0; l < cutFrameRectCopy.size().height; l++)
						{
							if (cutFrameRectCopy.at<uchar>(l, k) == 255)
							{
								countWhiteLength++;
							}
						}
						if (countWhiteLength < minWhiteLength)
						{
							minWhiteLength = countWhiteLength;
							cutX = k;
						}
					}
					//找到白色最少的地方，切割
					cutEdge.push_back(cutX);
				}
				cutEdge.push_back(cutFrameRectCopy.size().width - 1);
				//所有边已被存储，开始分割图像


				for (int j = 0; j < cutEdge.size() - 1; j++)
				{
					skeleton PersonTempSkeleton;
					//开始处理每张图的情况
					cv::Mat PersonTempMat, PersonTempMatCopy, PersonTempThin, PersonTempMats;

					cv::Rect PersonTempRect = cv::Rect(cutEdge[j], 0, cutEdge[j + 1] - cutEdge[j] + 1, 150);
					cutFrameRectCopy(PersonTempRect).copyTo(PersonTempMat);

					//PersonTempMat = Mat::zeros(PersonTempMats.size(), CV_8UC1);
					//RemoveSmallRegion(PersonTempMats, PersonTempMat, 50, 1, 0);

					vector<vector<cv::Point>> PersonTempContours;
					PersonTempMat.copyTo(PersonTempMatCopy);


					findContours(PersonTempMatCopy, PersonTempContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
					std::sort(PersonTempContours.begin(), PersonTempContours.end(), sortCountersArea);

					cv::Moments mu = moments(PersonTempContours[0]);
					if (mu.m00 == 0)
					{
						continue;
					}
					else
					{
						//存储每张图片的重心点
						cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


						//计算身体宽度，骨架化，裁剪最后一行

						PersonTempThin = thinImage(PersonTempMat);

						for (int k = 0; k < PersonTempThin.size().width; k++)
						{
							PersonTempThin.at<uchar>(cv::Point(k, PersonTempThin.size().height - 1)) = 0;
						}

						for (int k = 0; k < PersonTempThin.size().height; k++)
						{
							PersonTempThin.at<uchar>(cv::Point(0, k)) = 0;
						}

						for (int k = 0; k < PersonTempThin.size().height; k++)
						{
							PersonTempThin.at<uchar>(cv::Point(PersonTempThin.size().width - 1, k)) = 0;
						}


						//算出各个点
						vector<cv::Point2f> endpoints = skeletonEndPoints(PersonTempThin);
						vector<cv::Point2f> branchpoints = skeletonBranchPoints(PersonTempThin, 4, 5, 4);

						skeleton skeletonData = FromEdgePoints(endpoints, branchpoints, mc, PersonTempMat);

						//对skeletonData进行处理，换算到全局单位
						if (skeletonData._heart != cv::Point2f(0, 0))
							PersonTempSkeleton._heart = cv::Point(round((skeletonData._heart.x + PersonTempRect.x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData._heart.y + PersonTempRect.y) / (150.0 / PersonRect.height) + PersonRect.y));

						for (int k = BodyData_head; k != BodyData_len; k++)
						{
							if (skeletonData.bodyPoint[k] != cv::Point2f(0, 0))
								PersonTempSkeleton.bodyPoint[k] = cv::Point(round((skeletonData.bodyPoint[k].x + PersonTempRect.x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData.bodyPoint[k].y + PersonTempRect.y) / (150.0 / PersonRect.height) + PersonRect.y));

						}

						//将轮廓映射到全画面
						vector<cv::Point> skeletonContours;
						for (auto& k : PersonTempContours[0])
						{
							cv::Point ContoursPointTemp;
							ContoursPointTemp = cv::Point(round((k.x + PersonTempRect.x) / (150.0 / PersonRect.height) + PersonRect.x), round((
								                              k.y + PersonTempRect.y) / (150.0 / PersonRect.height) + PersonRect.y));
							skeletonContours.push_back(ContoursPointTemp);
						}

						PersonTempSkeleton.skeletonContours = skeletonContours;

						PersonSkeleton.push_back(PersonTempSkeleton);
					}

				}
			}
			else
			{

				//处理单人的情况
				skeleton PersonTempSkeleton;
				cv::Mat PersonTempMat, PersonTempMatCopy, PersonTempThin;
				cutFrameRectCopy.copyTo(PersonTempMat);

				vector<vector<cv::Point>> PersonTempContours;
				PersonTempMat.copyTo(PersonTempMatCopy);


				findContours(PersonTempMatCopy, PersonTempContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
				std::sort(PersonTempContours.begin(), PersonTempContours.end(), sortCountersArea);

				cv::Moments mu = moments(PersonTempContours[0]);
				if (mu.m00 == 0)
				{
					continue;
				}
				else
				{
					//存储每张图片的重心点
					cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


					//计算身体宽度，骨架化，裁剪最后一行

					PersonTempThin = thinImage(PersonTempMat);

					for (int i = 0; i < PersonTempThin.size().width; i++)
					{
						PersonTempThin.at<uchar>(cv::Point(i, PersonTempThin.size().height - 1)) = 0;
					}

					for (int i = 0; i < PersonTempThin.size().height; i++)
					{
						PersonTempThin.at<uchar>(cv::Point(0, i)) = 0;
						PersonTempThin.at<uchar>(cv::Point(PersonTempThin.size().width - 1, i)) = 0;
					}


					//算出各个点
					vector<cv::Point2f> endpoints = skeletonEndPoints(PersonTempThin);
					vector<cv::Point2f> branchpoints = skeletonBranchPoints(PersonTempThin, 4, 5, 4);


					skeleton skeletonData = FromEdgePoints(endpoints, branchpoints, mc, PersonTempMat);

					//对skeletonData进行处理，换算到全局单位
					if (skeletonData._heart != cv::Point2f(0, 0))
						PersonTempSkeleton._heart = cv::Point(round((skeletonData._heart.x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData._heart.y) / (150.0 / PersonRect.height) + PersonRect.y));

					for (int k = BodyData_head; k != BodyData_len; k++)
					{
						if (skeletonData.bodyPoint[k] != cv::Point2f(0, 0))
							PersonTempSkeleton.bodyPoint[k] = cv::Point(round((skeletonData.bodyPoint[k].x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData.bodyPoint[k].y) / (150.0 / PersonRect.height) + PersonRect.y));

					}

					vector<cv::Point> skeletonContours;
					for (auto& k : PersonTempContours[0])
					{
						cv::Point ContoursPointTemp;
						ContoursPointTemp = cv::Point(round((k.x) / (150.0 / PersonRect.height) + PersonRect.x), round((
							                              k.y) / (150.0 / PersonRect.height) + PersonRect.y));
						skeletonContours.push_back(ContoursPointTemp);
					}

					PersonTempSkeleton.skeletonContours = skeletonContours;
					PersonSkeleton.push_back(PersonTempSkeleton);
				}
			}
			std::cout << i << ":" << area << " " << cutFrameRectCopy.size().width << endl;
		}

		//完成识别，开始对当前骨架分配Index
		vector<PersonData> oldPersonInformation = PersonInformation;
		vector<skeleton> tempPersonSkeleton = PersonSkeleton;
		for (auto& i : PersonSkeleton)
		{
			PersonData ClosestPerson;
			double MinCenterDistance = 999;
			for (auto& j : oldPersonInformation)
			{
				const double CenterDistance = sqrt(pow(i._heart.x - j.skeletonData._heart.x, 2) + pow(i._heart.y - j.skeletonData._heart.y, 2));
				if (CenterDistance < MinCenterDistance && CenterDistance < 100)
				{
					MinCenterDistance = CenterDistance;
					ClosestPerson = j;
				}
			}

			if (ClosestPerson.index != -1)
			{
				//表示找到了最近的那个人
				auto itold = find(oldPersonInformation.begin(), oldPersonInformation.end(), ClosestPerson);
				auto itnow = find(PersonInformation.begin(), PersonInformation.end(), ClosestPerson);
				if (itold != oldPersonInformation.end())
				{
					if (itnow->oldSskeletonData.size() < 10)
					{
						itnow->oldSskeletonData.push_front(itnow->skeletonData);
						itnow->skeletonData = i;

					}
					else
					{
						itnow->oldSskeletonData.pop_back();
						itnow->oldSskeletonData.push_front(itnow->skeletonData);
						itnow->skeletonData = i;
					}

					auto iterator = find(tempPersonSkeleton.begin(), tempPersonSkeleton.end(), i);
					if (iterator != tempPersonSkeleton.end())
					{
						tempPersonSkeleton.erase(iterator);
					}
					oldPersonInformation.erase(itold);
				}
			}
		}


		if (!oldPersonInformation.empty())
		{
			//有人没分配到新动作，可能已经下场
			//将其从队列中删掉
			for (const auto& i : oldPersonInformation)
			{
				auto iterator = find(PersonInformation.begin(), PersonInformation.end(), i);
				if (iterator != PersonInformation.end())
				{
					PersonInformation.erase(iterator);
				}
			}
		}


		if (!tempPersonSkeleton.empty())
		{
			//表示还有骨架没有分配完，可能有人新入场
			PersonData tempPersonData;
			for (const auto& i : tempPersonSkeleton)
			{
				tempPersonData.index = indexNum;
				tempPersonData.skeletonData = i;
				indexNum++;
				//防止经过的人太多
				if (indexNum > 30000)
					indexNum = 0;
				const DWORD endTime = GetCurrentTime();
				tempPersonData.m_fTimes = (endTime - startTime) / 1000.0;
				PersonInformation.push_back(tempPersonData);
			}
		}
	}
	else
	{
		return false;
	}
	return true;
}


void  CJcCalBody::GetBodyData(std::vector<BodyData> &BodyArr)
{
	std::vector<BodyData> BodyArrTemp;
	if (!PersonInformation.empty())
	{
		for (auto& i : PersonInformation)
		{
			BodyData Temp;
			Temp.m_contours = i.skeletonData.skeletonContours;
			Temp.m_fTimes = i.m_fTimes;
			Temp._heart = i.skeletonData._heart;
			Temp._index = i.index;

			for (int j = BodyData_head; j < BodyData_len; j++)
			{
				shared_ptr<jcBlockData> BlockDataTemp = std::make_shared<jcBlockData>();
				if (i.skeletonData.bodyPoint[j] != cv::Point2f(0, 0))
				{

					BlockDataTemp->pos = i.skeletonData.bodyPoint[j];

					if (j < 3)
					{
						BlockDataTemp->dir = cv::Point(i.skeletonData.bodyPoint[j].x - i.skeletonData.bodyPoint[BodyData_chest].x,
						                               i.skeletonData.bodyPoint[j].y - i.skeletonData.bodyPoint[BodyData_chest].y);
					}
					else
					{
						BlockDataTemp->dir = cv::Point(0, 0);
					}
					Temp._keyBodyDts[j].push_back(BlockDataTemp);
				}
				else
				{
					Temp._keyBodyDts[j].push_back(nullptr);
				}
			}
			BodyArrTemp.push_back(Temp);
		}
	}
	BodyArr = BodyArrTemp;
}