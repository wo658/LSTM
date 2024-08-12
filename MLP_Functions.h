#pragma once

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include "csvparser.h"


using namespace std;


vector<vector<double>> readFile(const char* aNameOfDataWithDir); 

//PreProcessing
void splitData(vector<vector<double>> rawData, vector<vector<double>> &x, vector<vector<double>> &y);
