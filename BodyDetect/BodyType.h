#pragma once
#include "BodyType.h"
#include <cv.hpp>
#include <iostream>
#include<deque>

using namespace std;


enum Enum_BodyData
{
	BodyData_head,//
	BodyData_leftHand,
	BodyData_rightHand,
	BodyData_leftFoot,
	BodyData_rightFoot,
	BodyData_chest,//��
	BodyData_hip,//��
	BodyData_len
};

struct skeleton
{
	cv::Point2f bodyPoint[BodyData_len] = { cv::Point2f(0, 0) };
	cv::Point2f _heart = cv::Point2f(0, 0);
	bool operator == (const skeleton &i) const;
	vector<cv::Point> skeletonContours;
};

struct PersonData
{
	int index = -1;
	float m_fTimes;//ʶ���ʱ��
	skeleton skeletonData;
	deque<skeleton> oldSskeletonData;
	bool operator == (const PersonData &i) const;
};



class jcBlockData
{
public:
	cv::Point dir;//����
	cv::Point pos;//λ��
	void operator=(jcBlockData a);
};

class BodyData
{
public:
	unsigned int _index;
	std::vector<shared_ptr<jcBlockData>> _keyBodyDts[BodyData_len];//���ؼ��㣬δʶ�������ΪNULL
	float m_fTimes;//ʶ���ʱ��

	cv::Point _heart;//����
	std::vector<cv::Point> m_contours;//����
};

struct TornadoData
{
	int _index;
	Enum_BodyData _type;

	cv::Point _pos;
};