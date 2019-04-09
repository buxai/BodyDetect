#include "Body.h"
#include "BodyType.h"
#include "BodyDetect.h"
#include "ImageSegmentation.h"

bool CJcCalBody::recognizeImage(cv::Mat& img)
{
	//�����Ѷ�ֵ�����ü����ĺڰ�ͼ

	DWORD startTime = GetCurrentTime();
	cv::Mat imgCopy;
	img.copyTo(imgCopy);
	vector<vector<cv::Point> > contours;
	findContours(imgCopy, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	std::sort(contours.begin(), contours.end(), sortCountersArea);

	//ɾ�������С������
	vector<vector<cv::Point>>::iterator it;
	for (it = contours.begin(); it != contours.end();)
	{
		if (contourArea(*it) < 1000)
		{
			it = contours.erase(it);
		}
		else
		{
			it++;
		}
	}

	if (contours.size() > 0)
	{

		vector<skeleton> PersonSkeleton;	//ÿ���˶�Ӧ�ĹǼ�
		for (int i = 0; i < contours.size(); i++)
		{
			vector<cv::Mat> PersonMat;	//ÿ���˶�Ӧ��Mat
			cv::Rect PersonRect = boundingRect(contours[i]);

			cv::Mat cutFrameRectCopy;
			img(PersonRect).copyTo(cutFrameRectCopy);


			resize(cutFrameRectCopy, cutFrameRectCopy, cv::Size(150.0*PersonRect.width / PersonRect.height, 150.0));

			int Area = contourArea(contours[i]) * pow((150.0 / PersonRect.height), 2);
			int PersonNumber = min(Area / 4000.0, cutFrameRectCopy.size().width / 52.5);



			if (PersonNumber >= 2)
			{

				//��ʾ��ǰRect�������ܶ���1�ˣ���Ҫ�ü�
				int PersonWidth = cutFrameRectCopy.size().width / PersonNumber;	//�õ�ÿ���˵Ĵ�ſ��


				vector<cv::Point> heightPoint;	//ÿ���˵�����
				for (int j = 0; j < PersonNumber; j++)
				{
					//����ÿ��ͼ
					cv::Rect cutPersonRect(j*PersonWidth, 0, PersonWidth, cutFrameRectCopy.size().height);
					cv::Mat PersonTemp;
					cutFrameRectCopy(cutPersonRect).copyTo(PersonTemp);	//�洢ÿ���˵���ʱͼ��

																		//���ÿ��ͼƬ������
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
						//�洢ÿ��ͼƬ�����ĵ�
						cv::Point mc = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
						heightPoint.push_back(cv::Point(j*PersonWidth + mc.x, mc.y));
						//circle(cutFrameRectCopy, Point(j*PersonWidth + mc.x, mc.y), 2, Scalar(0, 0, 0));
					}
				}



				vector<int> cutEdge;
				cutEdge.push_back(0);

				//Ѱ������ͼƬ֮����Ҫ�и�ĵط�
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
					//�ҵ���ɫ���ٵĵط����и�
					cutEdge.push_back(cutX);

					/*
					for (int l = 0; l < cutFrameRectCopy.size().height; l++)
					{
					cutFrameRectCopy.at<uchar>(l, cutX) = 0;
					}
					imshow("test", cutFrameRectCopy);
					*/
				}
				cutEdge.push_back(cutFrameRectCopy.size().width - 1);
				//���б��ѱ��洢����ʼ�ָ�ͼ��


				for (int j = 0; j < cutEdge.size() - 1; j++)
				{
					skeleton PersonTempSkeleton;
					//��ʼ����ÿ��ͼ�����
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
						//�洢ÿ��ͼƬ�����ĵ�
						cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


						//���������ȣ��Ǽܻ����ü����һ��

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


						//���������
						vector<cv::Point2f> endpoints = skeletonEndPoints(PersonTempThin);
						vector<cv::Point2f> branchpoints = skeletonBranchPoints(PersonTempThin, 4, 5, 4);

						skeleton skeletonData = FromEdgePoints(endpoints, branchpoints, mc, PersonTempMat, PersonTempContours[0]);

						//��skeletonData���д������㵽ȫ�ֵ�λ
						if (skeletonData._heart != cv::Point2f(0, 0))
							PersonTempSkeleton._heart = cv::Point(round((skeletonData._heart.x + PersonTempRect.x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData._heart.y + PersonTempRect.y) / (150.0 / PersonRect.height) + PersonRect.y));

						for (int k = BodyData_head; k != BodyData_len; k++)
						{
							if (skeletonData.bodyPoint[k] != cv::Point2f(0, 0))
								PersonTempSkeleton.bodyPoint[k] = cv::Point(round((skeletonData.bodyPoint[k].x + PersonTempRect.x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData.bodyPoint[k].y + PersonTempRect.y) / (150.0 / PersonRect.height) + PersonRect.y));

						}

						//������ӳ�䵽ȫ����
						vector<cv::Point> skeletonContours;
						for (int k = 0; k < PersonTempContours[0].size(); k++)
						{
							cv::Point ContoursPointTemp;
							ContoursPointTemp = cv::Point(round((PersonTempContours[0][k].x + PersonTempRect.x) / (150.0 / PersonRect.height) + PersonRect.x), round((PersonTempContours[0][k].y + PersonTempRect.y) / (150.0 / PersonRect.height) + PersonRect.y));
							skeletonContours.push_back(ContoursPointTemp);
						}

						PersonTempSkeleton.skeletonContours = skeletonContours;

						PersonSkeleton.push_back(PersonTempSkeleton);

						/*
						circle(videoDisplay, PersonTempSkeleton.Center, 4, Scalar(255, 0, 255), -1);

						circle(videoDisplay, PersonTempSkeleton.head, 4, Scalar(255, 0, 0), -1);
						circle(videoDisplay, PersonTempSkeleton.lefthand, 4, Scalar(0, 255, 0), -1);
						circle(videoDisplay, PersonTempSkeleton.righthand, 4, Scalar(0, 0, 255), -1);

						circle(videoDisplay, PersonTempSkeleton.chest, 4, Scalar(256, 128, 0), -1);
						circle(videoDisplay, PersonTempSkeleton.hip, 4, Scalar(0, 128, 256), -1);

						circle(videoDisplay, PersonTempSkeleton.leftfoot, 4, Scalar(255, 255, 0), -1);
						circle(videoDisplay, PersonTempSkeleton.rightfoot, 4, Scalar(0, 255, 255), -1);
						*/
					}

				}



			}
			else
			{

				//�����˵����
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
					//�洢ÿ��ͼƬ�����ĵ�
					cv::Point2f mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);


					//���������ȣ��Ǽܻ����ü����һ��

					PersonTempThin = thinImage(PersonTempMat);

					for (int i = 0; i < PersonTempThin.size().width; i++)
					{
						PersonTempThin.at<uchar>(cv::Point(i, PersonTempThin.size().height - 1)) = 0;
					}

					for (int i = 0; i < PersonTempThin.size().height; i++)
					{
						PersonTempThin.at<uchar>(cv::Point(0, i)) = 0;
					}

					for (int i = 0; i < PersonTempThin.size().height; i++)
					{
						PersonTempThin.at<uchar>(cv::Point(PersonTempThin.size().width - 1, i)) = 0;
					}


					//���������
					vector<cv::Point2f> endpoints = skeletonEndPoints(PersonTempThin);
					vector<cv::Point2f> branchpoints = skeletonBranchPoints(PersonTempThin, 4, 5, 4);


					skeleton skeletonData = FromEdgePoints(endpoints, branchpoints, mc, PersonTempMat, PersonTempContours[0]);

					//��skeletonData���д������㵽ȫ�ֵ�λ
					if (skeletonData._heart != cv::Point2f(0, 0))
						PersonTempSkeleton._heart = cv::Point(round((skeletonData._heart.x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData._heart.y) / (150.0 / PersonRect.height) + PersonRect.y));

					for (int k = BodyData_head; k != BodyData_len; k++)
					{
						if (skeletonData.bodyPoint[k] != cv::Point2f(0, 0))
							PersonTempSkeleton.bodyPoint[k] = cv::Point(round((skeletonData.bodyPoint[k].x) / (150.0 / PersonRect.height) + PersonRect.x), round((skeletonData.bodyPoint[k].y) / (150.0 / PersonRect.height) + PersonRect.y));

					}

					vector<cv::Point> skeletonContours;
					for (int k = 0; k < PersonTempContours[0].size(); k++)
					{
						cv::Point ContoursPointTemp;
						ContoursPointTemp = cv::Point(round((PersonTempContours[0][k].x) / (150.0 / PersonRect.height) + PersonRect.x), round((PersonTempContours[0][k].y) / (150.0 / PersonRect.height) + PersonRect.y));
						skeletonContours.push_back(ContoursPointTemp);
					}

					PersonTempSkeleton.skeletonContours = skeletonContours;
					PersonSkeleton.push_back(PersonTempSkeleton);



					/*
					circle(videoDisplay, PersonTempSkeleton.Center, 4, Scalar(255, 0, 255), -1);

					circle(videoDisplay, PersonTempSkeleton.head, 4, Scalar(255, 0, 0), -1);
					circle(videoDisplay, PersonTempSkeleton.lefthand, 4, Scalar(0, 255, 0), -1);
					circle(videoDisplay, PersonTempSkeleton.righthand, 4, Scalar(0, 0, 255), -1);

					circle(videoDisplay, PersonTempSkeleton.chest, 4, Scalar(256, 128, 0), -1);
					circle(videoDisplay, PersonTempSkeleton.hip, 4, Scalar(0, 128, 256), -1);

					circle(videoDisplay, PersonTempSkeleton.leftfoot, 4, Scalar(255, 255, 0), -1);
					circle(videoDisplay, PersonTempSkeleton.rightfoot, 4, Scalar(0, 255, 255), -1);
					*/
				}
				//PersonMat.push_back(cutFrameRectCopy);
			}
			std::cout << i << ":" << Area << " " << cutFrameRectCopy.size().width << endl;
		}

		//���ʶ�𣬿�ʼ�Ե�ǰ�Ǽܷ���Index
		vector<PersonData> oldPersonInformation = PersonInformation;
		vector<skeleton> tempPersonSkeleton = PersonSkeleton;
		for (int i = 0; i < PersonSkeleton.size(); i++)
		{
			PersonData ClosestPerson;
			double MinCenterDistance = 999;
			for (int j = 0; j < oldPersonInformation.size(); j++)
			{
				double CenterDistance = sqrt(pow(PersonSkeleton[i]._heart.x - oldPersonInformation[j].skeletonData._heart.x, 2) + pow(PersonSkeleton[i]._heart.y - oldPersonInformation[j].skeletonData._heart.y, 2));
				if (CenterDistance < MinCenterDistance && CenterDistance < 100)
				{
					MinCenterDistance = CenterDistance;
					ClosestPerson = oldPersonInformation[j];
				}
			}

			if (ClosestPerson.index != -1)
			{
				//��ʾ�ҵ���������Ǹ���
				vector<PersonData>::iterator itold = find(oldPersonInformation.begin(), oldPersonInformation.end(), ClosestPerson);
				vector<PersonData>::iterator itnow = find(PersonInformation.begin(), PersonInformation.end(), ClosestPerson);
				if (itold != oldPersonInformation.end())
				{
					if (itnow->oldSskeletonData.size() < 10)
					{
						itnow->oldSskeletonData.push_front(itnow->skeletonData);
						itnow->skeletonData = PersonSkeleton[i];

					}
					else
					{
						itnow->oldSskeletonData.pop_back();
						itnow->oldSskeletonData.push_front(itnow->skeletonData);
						itnow->skeletonData = PersonSkeleton[i];
					}

					vector<skeleton>::iterator it = find(tempPersonSkeleton.begin(), tempPersonSkeleton.end(), PersonSkeleton[i]);
					if (it != tempPersonSkeleton.end())
					{
						tempPersonSkeleton.erase(it);
						int a = 0;
					}
					oldPersonInformation.erase(itold);
					int a = 0;
				}
			}
		}


		if (oldPersonInformation.size() > 0)
		{
			//����û���䵽�¶����������Ѿ��³�
			//����Ӷ�����ɾ��
			for (int i = 0; i < oldPersonInformation.size(); i++)
			{
				vector<PersonData>::iterator it = find(PersonInformation.begin(), PersonInformation.end(), oldPersonInformation[i]);
				if (it != PersonInformation.end())
				{
					PersonInformation.erase(it);
				}
			}
		}


		if (tempPersonSkeleton.size() > 0)
		{
			//��ʾ���йǼ�û�з����꣬�����������볡
			PersonData tempPersonData;
			for (int i = 0; i < tempPersonSkeleton.size(); i++)
			{
				tempPersonData.index = indexNum;
				tempPersonData.skeletonData = tempPersonSkeleton[i];
				indexNum++;
				//��ֹ��������̫��
				if (indexNum > 30000)
					indexNum = 0;
				DWORD endTime = GetCurrentTime();
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
	jcBlockData* BlockDataTemp;
	std::vector<BodyData> BodyArrTemp;
	if (PersonInformation.size() > 0)
	{
		for (int i = 0; i < PersonInformation.size(); i++)
		{
			BodyData Temp;
			Temp.m_contours = PersonInformation[i].skeletonData.skeletonContours;
			Temp.m_fTimes = PersonInformation[i].m_fTimes;
			Temp._heart = PersonInformation[i].skeletonData._heart;
			Temp._index = PersonInformation[i].index;

			for (int j = BodyData_head; j < BodyData_len; j++)
			{
				BlockDataTemp = new jcBlockData();
				if (PersonInformation[i].skeletonData.bodyPoint[j] != cv::Point2f(0, 0))
				{

					BlockDataTemp->pos = PersonInformation[i].skeletonData.bodyPoint[j];

					if (j < 3)
					{
						BlockDataTemp->dir = cv::Point(PersonInformation[i].skeletonData.bodyPoint[j].x - PersonInformation[i].skeletonData.bodyPoint[BodyData_chest].x,
							PersonInformation[i].skeletonData.bodyPoint[j].y - PersonInformation[i].skeletonData.bodyPoint[BodyData_chest].y);
					}
					else
					{
						BlockDataTemp->dir = cv::Point(0, 0);
					}
					Temp._keyBodyDts[j].push_back(BlockDataTemp);
				}
				else
				{
					Temp._keyBodyDts[j].push_back(NULL);
				}
			}
			BodyArrTemp.push_back(Temp);
		}
	}
	BodyArr = BodyArrTemp;
}