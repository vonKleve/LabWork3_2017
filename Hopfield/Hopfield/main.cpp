//#include "Network.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include "Network.h"

using namespace std;
using namespace network;

void show(uchar**, int);
template<typename T>
void show(vector<vector<T>>&);
template<typename T>
void show(vector<vector<T>>&,int);

int main()
{
	int exSize = 28, exImageAmount = 100, exLabelsAmount = 100, amountOfWorkingEx = 10;
	int dataAmount = 0, dataLabelsAmount = 0, dataSize = 0, amountOfWorkingData = 5;

	string exDataPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\t10k-images.idx3-ubyte";
	string exLabelsPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\t10k-labels.idx1-ubyte";
	string dataPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\train-images.idx3-ubyte";
	string dataLabelsPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\train-labels.idx1-ubyte";

	uchar** example = read_mnist_images(exDataPath, exImageAmount, exSize);
	uchar* exampleLabels = read_mnist_labels(exLabelsPath, exLabelsAmount);
	uchar** dataIm = read_mnist_images(dataPath, dataAmount, dataSize);
	uchar* dataLabels = read_mnist_labels(dataLabelsPath, dataLabelsAmount);


	cout << "Imported images - patterns: " << exImageAmount << " and imported labels: " << exLabelsAmount << ". 1 image contains: " << exSize << " pixels." << endl;
	cout << "Imported: " << dataAmount << " and labels: " << dataLabelsAmount << " . 1 image contains:" << dataSize << " pixels" << endl;

	vector<vector<int>>exdata;
	vector<vector<int>>data;
	vector<vector<double>>weights;

	cout << "Amounts of separated examples: " << amountOfWorkingEx << " , data: " << amountOfWorkingData << endl;

	SeparateData(example, exdata, amountOfWorkingEx, exSize);
	SeparateData(dataIm, data, amountOfWorkingData, dataSize);

	// +++++++++++++++++++++++++++++++++ STATISTICS ++++++++++++++++++++++++++++++++++++++++++++++++
	int dataStat[10] = { 0 };
	int exStat[10] = { 0 };
	for (int i = 0; i < amountOfWorkingEx; i++)
	{
		dataStat[(int)dataLabels[i]]++;
		exStat[(int)exampleLabels[i]]++;
	}
	cout << "Example statics: ";
	for (int i = 0; i < 10; i++)
	{
		cout << i << ".(" << exStat[i] << ")";
	}
	cout << endl;
	cout << "Data statistics: ";
	for (int i = 0; i < 10; i++)
	{
		cout << i << ".(" << dataStat[i] << ")";
	}
	cout << endl;
	// +++++++++++++++++++++++++++++++++++++++++++ Learning ++++++++++++++++++++++++++++++++++++++++++++++++++++++

	cout << "Learning...\n";
	unsigned int start_time = clock();

	LearnMatrix(exdata, weights, exSize);

	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	cout << "Learning complete! Working time: " << search_time << endl;


	// +++++++++++++++++++++++++++++++++++++ Executing ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	for (int i = 0; i < amountOfWorkingData; i++)
	{
		cout << "Executing...\n";
		cout << "Before: \n";
		show(data, i);
		cout << endl;

		Execute(data[i], weights, dataSize);

		cout << "After: \n";
		show(data, i);
		cout << "\n\n";
	}
	cout << "End.\n";


	for (int i = 0; i < amountOfWorkingEx; i++)
	{
		delete[] example[i];
	}
	for (int i = 0; i < amountOfWorkingData; i++)
	{
		delete[] dataIm[i];
	}
	delete[] example;
	delete[] dataIm;
	delete[] dataLabels;
	delete[] exampleLabels;

	system("pause");
	return 0;
}

void show(uchar **arr, int amount)
{
	for (int i = 0; i < amount; i++)
	{
		for (int j = 0; j < 784; j++)
		{
			if (j % 28 == 0) cout << endl;
			cout << arr[i][j];
		}
	}
}
template<typename T>
void show(vector<vector<T>>&arr)
{
	for (int i = 0; i < arr.size(); i++)
	{
		for (int j = 0; j < 784; j++)
		{
			if (j % 28 == 0) cout << endl;
			cout << arr[i][j];
		}
		cout << endl;
	}
}

template<typename T>
void show(vector<vector<T>> &arr, int index)
{
	if (index < 0)
		index = arr.size() - 1;
	if (index >= arr.size())
		index = arr.size() - 1;


	for (int j = 0; j < 784; j++)
	{
		if (j % 28 == 0) cout << endl;
		cout << arr[index][j];
	}
}