#include "disparityMap.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <ppl.h>
#include <algorithm>

disparityMap::disparityMap()
{
	m_iBlockSearchRange = 255;
	m_iBlockSize = 5;
}

disparityMap::disparityMap(int BlockSearchRange, int BlockSize)
	:m_iBlockSearchRange(BlockSearchRange),
	m_iBlockSize(BlockSize)
{
}


disparityMap::~disparityMap()
{
}
//Parallel version : Calculate the Disparity map in parallel mode
//imageLeft : The left image
//imageRight: The right image
//DisparityMap : The output disparity map
//void disparityMap::generateDisparityMapParallel(const Mat &imageLeft, const Mat &imageRight, Mat &disparityMap)
//{
//	Concurrency::parallel_for(int(0), imageLeft.rows - m_iBlockSize, [&](int row)
//	{
//		for (auto col = 0; col<imageLeft.cols - m_iBlockSize; col++)
//		{
//			cv::Mat blockToBeSearched(imageLeft, Rect(col, row, m_iBlockSize, m_iBlockSize));
//			auto diffVector = calculateBlocksDiffrencesInSearchRange(blockToBeSearched, row, col, imageRight);
//			//Find the minimum value and it's index.
//			auto minElement = min_element(begin(diffVector), end(diffVector));
//			auto ucIndex = static_cast<unsigned char>(distance(diffVector.begin(), minElement));
//			ucIndex = ucIndex + 1 > 255 ? 255 : ucIndex;
//			disparityMap.at<unsigned char>(row, col) = ucIndex + 1;
//			diffVector.clear();
//		}
//	});
//}

//Sequential version : Calculate the Disparity map in Sequential mode
//imageLeft : The left image
//imageRight: The right image
//DisparityMap : The output disparity map
//void disparityMap::generateDisparityMapSeq(const Mat& imageLeft, const Mat& imageRight, Mat& disparityMap)
//{
//	for (auto row = 0; row<imageLeft.rows - m_iBlockSize; row++)
//	{
//		for (auto col = 0; col<imageLeft.cols - m_iBlockSize; col++)
//		{
//			Mat blockToBeSearched(imageLeft, Rect(col, row, m_iBlockSize, m_iBlockSize));
//			auto diffVector = calculateBlocksDiffrencesInSearchRange(blockToBeSearched, row, col, imageRight);
//			//Find the minimum value and it's index.
//			auto minElement = min_element(begin(diffVector), end(diffVector));
//			auto ucIndex = static_cast<unsigned char>(distance(diffVector.begin(), minElement));
//			ucIndex = ucIndex + 1 > 255 ? 255 : ucIndex;
//			disparityMap.at<unsigned char>(row, col) = ucIndex + 1;
//		}
//	}
//}

//Compare blocks in a specific range. The compression is done on the right image from the position of - blockToBeSearched - to the left
// blockToBeSearched : The searched block from the left image
// row, col          : starting row and col of blockToBeSearched
// searchImage       : The right image. it's the image where a blocked is searched
//vector<unsigned int> disparityMap::calculateBlocksDiffrencesInSearchRange(const Mat &blockToBeSearched, int row, int col, Mat searchImage) const
//{
//	auto currentCol = col;
//	auto currentBlock = 1;
//	vector<unsigned int> blocksDiffrences;
//	//Searching to the left
//	while (currentBlock <= m_iBlockSearchRange && currentCol >= 0)
//	{
//		Mat searchImageBlock(searchImage, Rect(currentCol, row, m_iBlockSize, m_iBlockSize));
//		//compare blocks and returns the difference value
//		auto diff = CalculateBlockDifference(blockToBeSearched, searchImageBlock);
//		blocksDiffrences.push_back(diff);
//		--currentCol;
//		++currentBlock;
//	}
//
//	return blocksDiffrences;
//}

//blockToBeSearched : The searched block from the left image
//searchImageBlock  : The current block from the right image
int disparityMap::CalculateBlockDifference(const Mat & blockToBeSearched, const Mat & searchImageBlock) const
{
	unsigned int Sum = 0;
	for (auto row = 0; row < m_iBlockSize; row++)
	{
		for (auto col = 0; col < m_iBlockSize; col++)
		{
			//Sum = Sum + (searchImageBlock.at<unsigned char>(row, col) - blockToBeSearched.at<unsigned char>(row, col)) *
			//	(searchImageBlock.at<unsigned char>(row, col) - blockToBeSearched.at<unsigned char>(row, col));
			Sum = Sum + abs(searchImageBlock.at<unsigned char>(row, col) - blockToBeSearched.at<unsigned char>(row, col));
		}
	}
	return Sum;
}
//-------------------------------------------------------------
//Parallel version : Calculate the Disparity map in parallel mode
//imageLeft : The left image
//imageRight: The right image
//DisparityMap : The output disparity map
void disparityMap::generateDisparityMapParallel(const Mat &imageLeft, const Mat &imageRight, Mat &disparityMap)
{
	Concurrency::parallel_for(int(0), imageLeft.rows - m_iBlockSize, [&](int row)
	{
		for (auto col = 0; col<imageLeft.cols - m_iBlockSize; col++)
		{
			Mat blockToBeSearched(imageLeft, Rect(col, row, m_iBlockSize, m_iBlockSize));
			auto ucIndex = static_cast<unsigned char>(calculateBlocksDiffrencesInSearchRange(blockToBeSearched, row, col, imageRight));
			ucIndex = ucIndex + 1 > 255 ? 255 : ucIndex;
			disparityMap.at<unsigned char>(row, col) = ucIndex + 1;
		}
	});
}

//Sequential version : Calculate the Disparity map in Sequential mode
//imageLeft : The left image
//imageRight: The right image
//DisparityMap : The output disparity map
void disparityMap::generateDisparityMapSeq(const Mat& imageLeft, const Mat& imageRight, Mat& disparityMap)
{
	for (auto row = 0; row<imageLeft.rows - m_iBlockSize; row++)
	{
		for (auto col = 0; col<imageLeft.cols - (m_iBlockSize + 1); col++)
		{
			Mat blockToBeSearched(imageLeft, Rect(col, row, m_iBlockSize, m_iBlockSize));
			unsigned char ucIndex = static_cast<unsigned char>(calculateBlocksDiffrencesInSearchRange(blockToBeSearched, row, col, imageRight));
			ucIndex = ucIndex + 1 > 255 ? 255 : ucIndex;
			disparityMap.at<unsigned char>(row, col) = ucIndex + 1;
		}
	}
}

//This method calculates the difference between blockToBeSearched and the minimum found block. 
//I used the hierarchical search algorithm to find the minimum block
//The method returns the distance between the searchedblock and the one that is the most similiar to it.
unsigned int disparityMap::calculateBlocksDiffrencesInSearchRange(const Mat &blockToBeSearched, int row, int col, const Mat &searchImage) const
{
	auto currentCol = col;
	vector<int> blocksDiffrences;
	auto currentBlock = 1;
	auto blocks = m_iBlockSearchRange / m_iBlockSize;
	//This iteration find the most similiar block in block steps. after it will be find the next loop will search
	//around it 
	while (currentCol >= 0 && currentBlock <= blocks)
	{
		Mat searchImageBlock(searchImage, Rect(currentCol, row, m_iBlockSize, m_iBlockSize));
		//compare blocks and returns the difference value
		auto diff = CalculateBlockDifference(blockToBeSearched, searchImageBlock);
		blocksDiffrences.push_back(diff);
		currentCol = currentCol - m_iBlockSize;
		++currentBlock;
	}
	//The next use the best found block in the above iteration and search around it (1 block to the right to 1 block to the left)
	//for the best block
	auto minElement = min_element(begin(blocksDiffrences), end(blocksDiffrences));
	auto iIndex = distance(blocksDiffrences.begin(), minElement);
	auto colSearchStart = min(static_cast<int>(col - (m_iBlockSize * iIndex) + m_iBlockSize), static_cast<int>(searchImage.cols - (m_iBlockSize + 1)));
	auto colSearchEnd = col - (m_iBlockSize *(iIndex)) - m_iBlockSize;

	auto colMin = 0;
	auto max = 0;

	for (auto newCol = colSearchStart; (newCol >= colSearchEnd) && (newCol >= 0); newCol--)
	{
		Mat searchImageBlock(searchImage, Rect(newCol, row, m_iBlockSize, m_iBlockSize));
		auto diff = CalculateBlockDifference(blockToBeSearched, searchImageBlock);
		if (max == 0 || (max > 0 && diff < max))
		{
			max = diff;
			colMin = newCol;
		}

	}


	return abs(colMin-col);
}