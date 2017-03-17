#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

#define MAX_ITERATIONS 30

using namespace std;

namespace network
{
	//const unsigned int eps = 5;


	typedef unsigned char uchar;


	uchar** read_mnist_images(string, int&, int&);

	uchar* read_mnist_labels(string, int&);

	// Normalizing input data
	void SeparateData(uchar**, vector<vector<int>>&, const int, int);

	// returns normalized sum
	int GetSum(vector<double>&, vector<int>&);

	// training
	void LearnMatrix(vector<vector<int>>&, vector<vector<double>>&, int);

	// running
	void Execute(vector<int>&, vector<vector<double>>&, int);

	// find appropriate image label int exampleData for result vector
	//int FindAppropriate(vector<vector<int>> &exLabels, int dataLabel);
}