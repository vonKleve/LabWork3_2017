//#include "Network.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include "Network.h"

using namespace std;
using namespace network;

void show(uchar**, int);
void show(vector<vector<int>>&);
void show(vector<vector<int>>&,int = -1);
void show(vector < vector<double> >&);

int main()
{
	int imSize = 28, exImageAmount = 100, exLabelsAmount = 100, amountOfEx = 7;
	int dataAmount = 0, dataLabelsAmount = 0, dataSize = 0, amountOfData = 1;

	string exDataPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\t10k-images.idx3-ubyte";
	string exLabelsPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\t10k-labels.idx1-ubyte";
	string dataPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\train-images.idx3-ubyte";
	string dataLabelsPath = "C:\\Users\\Dell\\Desktop\\Give\\Data\\train-labels.idx1-ubyte";

	uchar** example = read_mnist_images(exDataPath, exImageAmount, imSize);
	uchar* exampleLabels = read_mnist_labels(exLabelsPath, exLabelsAmount);
	uchar** dataIm = read_mnist_images(dataPath, dataAmount, dataSize);
	uchar* dataLabels = read_mnist_labels(dataLabelsPath, dataLabelsAmount);


	cout << "Imported images - patterns: " << exImageAmount << " and imported labels: " << exLabelsAmount << ". 1 image contains: " << imSize << " pixels." << endl;
	cout << "Imported: " << dataAmount << " and labels: " << dataLabelsAmount << " . 1 image contains:" << dataSize << " pixels" << endl;


	vector<vector<int>>exdata;
	vector<vector<int>>data;
	vector<vector<double>>weights;

	cout << "Amounts of separated examples: " << amountOfEx << " , data: " << amountOfData << endl;

	SeparateData(example, exdata, amountOfEx);
	SeparateData(dataIm, data, amountOfData);

	// +++++++++++++++++++++++++++++++++ STATISTICS ++++++++++++++++++++++++++++++++++++++++++++++++
	int dataStat[10] = { 0 };
	int exStat[10] = { 0 };
	for (int i = 0; i < 100; i++)
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

	// +++++++++++++++++++++++++++++++++++++++++++
	cout << "Learning...\n";
	unsigned int start_time = clock();

	LearnMatrix(exdata, weights);

	unsigned int end_time = clock();
	unsigned int search_time = end_time - start_time;
	cout << "Learning complete! Working time: " << search_time << "\n";


	// +++++++++++++++++++++++++++++++++++++

	// @@@@@@@@@@@@@@@@@

	for (int i = 0; i < amountOfData; i++)
	{
		for (int j = 0; j < imSize; j++)
		{
			data[i][j] = exdata[i][j];
		}
	}

	// @@@@@@@@@@@@@@@@

	for (int i = 0; i < amountOfData; i++)
	{
		show(data, i);
		cout << endl;
		cout << "Executing...\n";

		Execute(data[i], weights);

		cout << "End.\n";
		show(data, i);
		cout << endl;
		cout << endl;
	}
	cout << "END OF WORK!\n\n\n";
	for (int i = 0; i < amountOfEx; i++)
	{
		delete[] example[i];
	}
	for (int i = 0; i < amountOfData; i++)
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

void show(vector<vector<int>>&arr)
{
	for (int i = 0; i < arr.size(); i++)
	{
		for (int j = 0; j < 784; j++)
		{
			if (j % 28 == 0) cout << endl;
			cout << arr[i][j];
		}
	}
}

void show(vector<vector<int>> &arr, int index)
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

void show(vector<vector<double>>&data)
{
	for (int i = 0; i < 784; i++)
	{
		for (int j = i; j < 784; j++)
		{
			cout << data[i][j];
		}
		cout << endl;
	}
}