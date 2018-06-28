//
// Created by yuanning on 18-6-28.
//

#include "lane_3d.h"
#include <vector>
#include <numeric>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
//#include "xtensor/xarray.hpp"
//#include "xtensor/xio.hpp"

using namespace std;
using namespace cv;

typedef struct Point {
    double x;
    double y;
    double z;
    Point (double x, double y, double z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
} Point;

class Lane3D {
public:
    Lane3D() {
        return;
    }

    int fit(vector<Point> &data, vector<Point> &curve, int displayed_points=100) {
        int ceil = 100;
        if (data.size() > ceil) {
            vector<unsigned int> indices(data.size());
            iota(indices.begin(), indices.end(), 0);
            random_shuffle(indices.begin(), indices.end());
            vector<Point> temp = vector<Point>();
            for (unsigned int i = 0; i < ceil; i++) {
                temp.push_back(data[indices[i]]);
            }
            data = temp;
//            for (unsigned int i = 0; i < data.size(); i++) {
//                Point p = data[i];
//                cout << i << " " << p.x << " " << p.y << " " << p.z << endl;
//            }
        }






        return 0;
    }
};

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

int read_off(string path, vector<Point> &lane_mask) {
    ifstream inFile;
    inFile.open(path);
    string title;
    inFile >> title;
//    cout << title << endl;
    int v, e, f;
    inFile >> v >> e >> f;
//    cout << v << " " << e << " " << f << endl;
    for (int i = 0; i < v; i++) {
        double x, y, z;
        inFile >> x >> y >> z;
        Point p = Point(x, y, z);
        lane_mask.push_back(p);
    }
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
//        cout << files[i] << endl;
        vector<Point> lane_mask = vector<Point>();
        read_off(dir + "/" + files[i], lane_mask);
//        for (unsigned int j = 0; j < lane_mask.size(); j++) {
//            Point p = lane_mask[j];
//            cout << p.x << " " << p.y << " " << p.z << endl;
//        }
        vector<Point> curve = vector<Point>();
        Lane3D lane_3d = Lane3D();
        lane_3d.fit(lane_mask, curve, 1000);
//        break;
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "Time used: " << duration << endl;
    return 0;
}
