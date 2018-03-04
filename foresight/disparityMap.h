#pragma once
#include <memory>
#include <vector>

namespace cv {
	class Mat;
}

using namespace cv;
using namespace std;

class disparityMap
{
public:
	//BlockSearchRane - The range to search for equivalent blocks
	//BlockSize - Block size
	disparityMap();
	disparityMap(int BlockSearchRange, int BlockSize);
	~disparityMap();

	void generateDisparityMapParallel(const Mat &imageLeft, const Mat &imageRight, Mat &disparityMap);
	void generateDisparityMapSeq(const Mat &imageLeft, const Mat &imageRight, Mat &disparityMap);
private:
	//vector<unsigned int> calculateBlocksDiffrencesInSearchRange(const Mat& blockToBeSearched, int row, int col, Mat searchImage) const;
	int CalculateBlockDifference(const Mat& blockToBeSearched, const Mat& currentBlock) const;

	//-----------------------------------
	unsigned int calculateBlocksDiffrencesInSearchRange(const Mat &blockToBeSearched, int row, int col, const Mat &searchImage) const;


private:
	int m_iBlockSearchRange;
	int m_iBlockSize;
};

