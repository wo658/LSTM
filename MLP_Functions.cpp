#include "MLP_Functions.h"

vector<vector<double>> readFile(const char* aNameOfDataWithDir) {
	vector<vector<double>>m_A;
	CsvParser *csvparser = CsvParser_new(aNameOfDataWithDir, ",", 1);
	CsvRow *row;

	const CsvRow *header = CsvParser_getHeader(csvparser);
	if (header == NULL) {
		printf("%s\n", CsvParser_getErrorMessage(csvparser));
	}
	m_A.clear();
	const char **headerFields = CsvParser_getFields(header);
	int numXfield = CsvParser_getNumFields(header);

	int TheCounter = 0;
	while ((row = CsvParser_getRow(csvparser))) {
		const char **rowFields = CsvParser_getFields(row);
		TheCounter++;
		//printf("Counting the number of Rows %d\n", TheCounter++);
		CsvParser_destroy_row(row);
	}

	CsvParser_destroy(csvparser);
	TheCounter = 0;

	const CsvRow *header2 = CsvParser_getHeader(csvparser);
	if (header2 == NULL) {
		//printf("%s\n", CsvParser_getErrorMessage(csvparser));
		/*		return 1;*/
	}

	csvparser = CsvParser_new(aNameOfDataWithDir, ",", 1);
	while ((row = CsvParser_getRow(csvparser))) {
		std::vector<double> temp;
		const char **rowFields = CsvParser_getFields(row);
		for (int ind = 0; ind < numXfield; ind++)
		{
			temp.push_back(strtod(rowFields[ind], NULL));

		}
		TheCounter++;
		m_A.push_back(temp);
		CsvParser_destroy_row(row);
	}

	CsvParser_destroy(csvparser);
	return m_A;
}

void splitData(vector<vector<double>> data, vector<vector<double>> &x, vector<vector<double>> &y)
{
	int rowLen = data.size();
	int colLen = data[0].size();
	x.resize(rowLen, vector<double>(colLen - 1, 0));
	y.resize(rowLen, vector<double>(1, 0));
	for (int i = 0; i < rowLen; i++) {
		for (int j = 0; j < colLen; j++) {
			if (j < colLen - 1)
				x[i][j] = data[i][j];
			else y[i][0] = data[i][j];
		}
	}
}

double determinant(vector<vector<double>> a)
{
	double determinant = 1;
	int k = 0;
	while (k < a.size() - 1)
	{
		for (int i = k; i < a.size(); i++)
		{
			if (a[k][k] != 0) {
				double c = a[i][k] / a[k][k];
				if (i != k) {
					for (int j = 0; j < a[0].size(); j++) {
						a[i][j] = a[i][j] - a[k][j] * c;
					}
				}
				else {
				}
			}
			else {
				if (i != k) {
					for (int j = 0; j < a[0].size(); j++) {
						a[i][j] = a[i][j] - a[i][k];
					}
				}
				else {
				}
			}
		}
		k++;
	}
	for (int i = 0; i < a.size(); i++) {
		determinant *= a[i][i];
	}

	return determinant;
}


