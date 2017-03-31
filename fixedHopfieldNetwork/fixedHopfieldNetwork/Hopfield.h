#pragma once

#include <iostream>
#include <vector>
#include <fstream>

#define MAX_ITERATIONS 100

using namespace std;

namespace network {
	extern int _data_used;
	extern int _examples_used;
	extern int _image_size;

	int ReverseInt(int);

	typedef unsigned char uchar;

	uchar** ReadMNISTImages(string, int&, int&);

	uchar* ReadMNISTLabels(string, int&);

	void ReadFile(std::string, int&, vector<vector<double>>&);

	void UCtovec(uchar**, int, vector<vector<double>>&);

	void Normalize(vector<double>&);

	void LearnMatrix(vector<vector<double>>&, vector<vector<double>>&);

	double NormalizedMultiply(vector<double>&, vector<double>&);

	void Execute(vector<vector<double>>&, vector<double>&);


	void show(vector<vector<double>>&, int);
	void show(vector<double>&);


	void input(vector<vector<double>>&, int);
}