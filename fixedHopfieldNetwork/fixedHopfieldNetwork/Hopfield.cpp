#include "Hopfield.h"
namespace network {

	int _image_size = 784;
	int _data_used = 1;
	int _examples_used = 1;

	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		const int c = 255;

		ch1 = i&c;
		ch2 = (i >> 8)&c;
		ch3 = (i >> 16)&c;
		ch4 = (i >> 24)&c;
		return (((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4);
	}

	uchar** ReadMNISTImages(string full_path, int& number_of_images, int& image_size) {
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

	uchar* ReadMNISTLabels(string full_path, int& number_of_labels) {
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

	void ReadFile(std::string path, int& items_quantity, vector<vector<double>>& output)
	{
		int image_size;
		char input;
		ifstream fin(path, ios_base::in);
		if (!fin.is_open())
		{
			cerr << "\nCannot open file!\n";
			return;
		}
		fin >> items_quantity;
		//items_quantity = (int)input;
		fin >> image_size;
		_image_size = image_size;

		output.resize(items_quantity);
		for (int i = 0; i < items_quantity; i++)
		{
			output[i].resize(image_size);
		}

		for (int i = 0; i < items_quantity; i++)
		{
			for (int j = 0; j < image_size; j++)
			{
				if (fin.eof())
				{
					cout << "ERROR ERROR ERROR.....";
					return;
				}
				fin >> output[i][j];
			}
		}
	}

	void UCtovec(uchar **input, int size, vector<vector<double>>&output)
	{
		output.resize(size);
		for (int i = 0; i < size; i++)
		{
			output[i].resize(_image_size);
		}

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < _image_size; j++)
			{
				if (input[i][j] > 0)
					output[i][j] = 1;
				else
					output[i][j] = -1;
			}
		}
	}


	void Normalize(vector<double>& vec)
	{
		for (int i = 0; i < _image_size; i++)
		{
			if (vec[i] >= 0)
				vec[i] = 1.;
			else
				vec[i] = -1.;
		}
	}

	void LearnMatrix(vector<vector<double>>& learned, vector<vector<double>>& patterns)
	{
		learned.resize(_image_size);
		for (int i = 0; i < _image_size; i++)
		{
			learned[i].resize(_image_size);
		}

		for (int i = 0; i < _image_size; i++)
		{
			for (int j = i + 1; j < _image_size; j++)
			{
				double koeff = 0;
				for (int k = 0; k < _examples_used; k++)
				{
					koeff += patterns[k][i] * patterns[k][j];
				}
				learned[i][j] = learned[j][i] = koeff;
			}
			learned[i][i] = 0;
		}
	}

	double NormalizedMultiply(vector<double>&op1, vector<double>&op2)
	{
		double result = 0;
		for (int i = 0; i < _image_size; i++)
		{
			result += op1[i] * op2[i];
		}
		if (result >= 0)
			return 1.;
		else
			return
			-1.;
	}

	void Execute(vector<vector<double>>&learned, vector<double>&data)
	{
		vector<double>ndata;
		ndata.resize(_image_size);

		int iteration_number = 0;
		bool changed = false;
		while (iteration_number < MAX_ITERATIONS && !changed)
		{
			for (int i = 0; i < _image_size; i++)
			{
				ndata[i] = NormalizedMultiply(learned[i], data);
			}
			iteration_number++;

			if (ndata == data)
				changed = true;
			data = ndata;
		}

	}

	void show(vector<vector<double>>& vec, int size)
	{
		int step = std::sqrt(_image_size);
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j <_image_size; j++)
			{
				if (j % step == 0)
					cout << endl;
				cout.width(2);
				cout << vec[i][j];
			}
			cout << endl;
		}
	}

	void show(vector<double>&vec)
	{
		int step = std::sqrt(_image_size);
		for (int i = 0; i <_image_size; i++)
		{
			if (i % step == 0)
				cout << endl;
			cout.width(2);
			cout << vec[i];
		}
		cout << endl;
	}

	void input(vector<vector<double>>&input, int size)
	{
		input.resize(size);
		for (int i = 0; i < size; i++)
		{
			input[i].resize(_image_size);
		}

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j <_image_size; j++)
			{
				cin >> input[i][j];
			}
			cout << endl << "NEXT:" << endl;
		}
	}
}