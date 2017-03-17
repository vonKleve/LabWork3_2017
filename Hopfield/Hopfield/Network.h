#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

using namespace std;

namespace network
{
	const unsigned int eps = 5;


	typedef unsigned char uchar;


	uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size);

	uchar* read_mnist_labels(string full_path, int& number_of_labels);

	void SeparateData(uchar**, vector<vector<int>>&lines, const int size);

	void LearnMatrix(vector<vector<int>>&, vector<vector<double>>&);

	void LearningCalc(vector<vector<int>>&, vector<vector<double>>&, int, int);

	void Execute(vector<int>&, vector<vector<double>>&);

	void Normalize(vector<int>&);

	//bool CheckEqual()
}