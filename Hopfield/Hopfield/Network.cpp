#include "Network.h"

namespace network
{
	uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size)
	{
		auto reverseInt = [](int i) {
			unsigned char c1, c2, c3, c4;
			c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
			return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		ifstream file(full_path, ios::binary);

		if (file.is_open()) {
			int magic_number = 0, n_rows = 0, n_cols = 0;

			file.read((char *)&magic_number, sizeof(magic_number));
			magic_number = reverseInt(magic_number);

			if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

			file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
			file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
			file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

			image_size = n_rows * n_cols;

			uchar** _dataset = new uchar*[number_of_images];
			for (int i = 0; i < number_of_images; i++) {
				_dataset[i] = new uchar[image_size];
				file.read((char *)_dataset[i], image_size);
			}
			return _dataset;
		}
		else {
			throw runtime_error("Cannot open file `" + full_path + "`!");
		}
	}

	uchar* read_mnist_labels(string full_path, int& number_of_labels) {
		auto reverseInt = [](int i) {
			unsigned char c1, c2, c3, c4;
			c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
			return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		ifstream file(full_path, ios::binary);

		if (file.is_open()) {
			int magic_number = 0;
			file.read((char *)&magic_number, sizeof(magic_number));
			magic_number = reverseInt(magic_number);

			if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

			file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

			uchar* _dataset = new uchar[number_of_labels];
			for (int i = 0; i < number_of_labels; i++) {
				file.read((char*)&_dataset[i], 1);
			}
			return _dataset;
		}
		else {
			throw runtime_error("Unable to open file `" + full_path + "`!");
		}
	}

	void SeparateData(uchar ** data, vector<vector<int>> &lines, const int size, int dataSize)
	{
		lines.resize(size);

		for (int i = 0; i < size; i++)
		{
			lines[i].resize(dataSize);
			for (int j = 0; j < 784; j++)
			{
				if (data[i][j] == 0) lines[i][j] = -1;
				else lines[i][j] = 1;
			}
		}
	}

	int GetSum(vector<double>& op1, vector<int>& op2)
	{
		double result = 0;
		for (int i = 0; i < op1.size(); i++)
		{
			result += op1[i] * op2[i];
		}
		if (result > 0)
			return 1;
		else return -1;
	}

	void LearnMatrix(vector<vector<int>> &data, vector<vector<double>> &toLearn, int imageSize)
	{
		toLearn.resize(imageSize);
		for (int i = 0; i < imageSize; i++)
		{
			toLearn[i].resize(imageSize);
		}

		double koeff = 0;
		for (int i = 0; i < imageSize; i++)
		{
			for (int j = i + 1; j < imageSize; j++)
			{
				for (int k = 0; k < data.size(); k++)
				{
					koeff += data[k][j] * data[k][i];
				}
				toLearn[i][j] = koeff;
				toLearn[j][i] = koeff;
				koeff = 0;
			}
			toLearn[i][i] = 0;
		}
	}

	void Execute(vector<int> &data, vector<vector<double>> &learnedMatrix, int imageSize)
	{
		vector<int> ndata;
		//vector<int>prevdata; - compair x + 1 and x - 1 data 
		ndata.resize(imageSize);
		double koeff = 0;
		bool flag = false;
		int counter = 0;
		
		while (!flag)
		{
			for (int i = 0; i < imageSize; i++)
			{
				koeff = GetSum(learnedMatrix[i], data);
				ndata[i] = koeff;
			}

			if (ndata == data)
			{
				flag = true;
			}
			data = ndata;

			counter++;
			if (counter >= MAX_ITERATIONS)
			{
				break;
			}
		}
	}
}