//Author: Yuanning Zhang (yuanningzhang@deepmotion.ai)
//Copyright (c) 2018-present, DeepMotion

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int getdir(string dir, vector<string> &files) {
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    sort(files.begin(), files.end());
    return 0;
}

int main(int argc, char *argv[]) {
    clock_t start;
    double duration;

    start = clock();

    string dir = "/home/yuanning/DeepMotion/pointclouds";
    vector<string> files = vector<string>();
    getdir(dir,files);

    for (unsigned int i = 2; i < files.size(); i++) {
        cout << files[i] << endl;
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "Time used: " << duration << endl;
    return 0;
}
