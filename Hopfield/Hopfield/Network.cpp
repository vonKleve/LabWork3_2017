#include "Network.h"


void show(vector<int>&arr)
{
		for (int j = 0; j < 784; j++)
		{
			if (j % 28 == 0) cout << endl;
			cout << arr[j];
		}
}


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

	void SeparateData(uchar ** data, vector<vector<int>> &lines, const int size)
	{
		lines.resize(size);

		for (int i = 0; i < size; i++)
		{
			lines[i].resize(784);
		}

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < 784; j++)
			{
				if (data[i][j] == 0) lines[i][j] = -1;
				else lines[i][j] = 1;
			}
		}
	}

	void LearnMatrix(vector<vector<int>> &data, vector<vector<double>> &toLearn)
	{
		toLearn.resize(784);
		for (int i = 0; i < 784; i++)
		{
			toLearn[i].resize(784);
		}


		/*std::thread thread1(LearningCalc, std::ref(data), std::ref(toLearn), 0, 195);
		thread1.join();
		std::thread thread2(LearningCalc, std::ref(data), std::ref(toLearn), 196, 392);
		thread2.join();
		std::thread thread3(LearningCalc, std::ref(data), std::ref(toLearn), 393, 588);
		thread3.join();
		std::thread thread4(LearningCalc, std::ref(data), std::ref(toLearn), 589, 784);
		thread4.join();*/

		for (int i = 0; i < 784; i++)
		{
			double koeff = 0;
			toLearn[i][i] = 0;
			for (int j = i + 1; j < 784; j++)
			{
				for (int k = 0; k < data.size(); k++)
				{
					koeff += data[k][j] * data[k][i];
				}
				toLearn[i][j] = koeff;
				toLearn[j][i] = koeff;
				koeff = 0;
			}
		}
	}


	void LearningCalc(vector<vector<int>> &data, vector<vector<double>> &toLearn, int downRange, int upperRange)
	{
		for (int i = downRange; i < upperRange; i++)
		{
			double koeff = 0;
			toLearn[i][i] = 0;
			for (int j = i + 1; j < 784; j++)
			{
				for (int k = 0; k < data.size(); k++)
				{
					koeff += data[k][j] * data[k][i];
				}
				toLearn[i][j] = toLearn[j][i] = koeff;
				koeff = 0;
			}
		}
	}

	void Execute(vector<int> &data, vector<vector<double>> &learnedMatrix)
	{
		vector<int> ndata;
		ndata.resize(784);
		double koeff = 0;
		bool flag = 0;
		int counter = 0;
		while (!flag)
		{
			for (int i = 0; i < 784; i++)
			{
				for (int j = 0; j < 784; j++)
				{
					koeff += learnedMatrix[j][i] * data[j];
				}
				ndata[i] = koeff;
				koeff = 0;
			}


			cout << "ndata: ";
			show(ndata);
			cout << endl << endl;
			Normalize(ndata);
			cout << "normalized data: ";
			show(ndata);
			cout << endl << endl;
			cout << "input: ";
			show(data);
			cout << endl << endl;


		//	if (data == ndata)
			//	flag = true;
			int delta = 0;
			for (int i = 0; i < 784; i++)
			{
				if (ndata[i] != data[i])
					delta++;
			}
			if (delta < eps)
				flag = true;

			for (int i = 0; i < 784; i++)
			{
				data[i] = ndata[i];
			}

			counter++;
			if (counter >= 30)
			{
				cout << "\nFailed!\n";
				break;
			}
		}

		cout << endl << counter << endl;
	}

	void Normalize(vector<int> &data)
	{
		for (int i = 0; i < data.size(); i++)
		{
			if (data[i] > 0)
				data[i] = 1;
			else if (data[i] < 0)
				data[i] = -1;
			else
				data[i] = 0;
		}
	}
}