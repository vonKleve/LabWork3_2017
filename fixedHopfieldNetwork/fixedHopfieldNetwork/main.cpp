#include <iostream>
#include "Hopfield.h"
#include <string>

using namespace std;
using namespace network;

void Run1();
void Run2();

int main()
{
	int choice;
	cin >> choice;
	switch (choice)
	{
	case 1:
		Run1();
		break;
	case 2:
		Run2();
		break;
	default:
		break;
	}
	system("pause");
	return 0;
}

void Run1()
{
	string example_path = "C:\\Users\\Dell\\Desktop\\Give\\Data\\t10k-images.idx3-ubyte";
	string example_labels_path = "C:\\Users\\Dell\\Desktop\\Give\\Data\\t10k-labels.idx1-ubyte";
	//string data_path = "C:\\Users\\Dell\\Desktop\\Give\\Data\\train-images.idx3-ubyte";
	std::string data_path = "MNISTImages.txt";
	string data_labels_path = "C:\\Users\\Dell\\Desktop\\Give\\Data\\train-labels.idx1-ubyte";
	int a, b;

	uchar** examplesUC = ReadMNISTImages(example_path, a, b);
	uchar* example_labelsUC = ReadMNISTLabels(example_labels_path, a);
	//uchar** dataUC = ReadMNISTImages(data_path, a, b);
	//uchar* data_labelsUC = ReadMNISTLabels(data_labels_path, a);

	vector<vector<double>>examples;
	vector<vector<double>>data;
	vector<vector<double>>learned;

	cout << "quantity of examples: ";
	cin >> _examples_used;
	UCtovec(examplesUC,_examples_used, examples);
	ReadFile(data_path, _data_used, data);
	//UCtovec(dataUC, DATA_USED, data);

	//input(data, DATA_USED);

	LearnMatrix(learned, examples);


	show(examples,examples.size());
	cout << endl << endl << endl << endl;
	show(data, data.size());
	cout << endl << endl << endl << endl;

	for (int i = 0; i < _data_used; i++)
	{
		cout << "BEFORE: " << endl;
		show(data[i]);
		Execute(learned, data[i]);
		cout << endl << "AFTER:" << endl;
		show(data[i]);
		cout << endl;
	}
}

void Run2()
{
	//std::string examples_path = "patterns.txt";
	//std::string data_path = "forms-ex.txt";
	//std::string examples_path = "MNISTexamples.txt";
	//std::string data_path = "MNISTImages.txt";
	std::string examples_path = "SimpleEx.txt";
	std::string data_path = "SimpleTest.txt";
	//std::string examples_path = "TESTExamples.txt";
	//std::string data_path = "TESTImages.txt";
	//cout << "\nExamples path: ";
	//cin >> examples_path;
	//cout << "\nData path: ";
	//cin >> data_path;
	//cout << endl;


	vector<vector<double>>examples;
	vector<vector<double>>data;
	vector<vector<double>>learned;

	ReadFile(examples_path, _examples_used, examples);
	ReadFile(data_path, _data_used ,data);

	cout << "Examples:\n";
	show(examples,examples.size());
	cout << endl << endl << "Not fixed DATA:\n" << endl;
	show(data, data.size());
	cout << endl << endl << endl;

	LearnMatrix(learned, examples);

	for (int i = 0; i < _data_used; i++)
	{
		Execute(learned, data[i]);
	}
	cout << "Fixed DATA:\n";
	show(data, data.size());
	cout << endl << endl << endl;
}