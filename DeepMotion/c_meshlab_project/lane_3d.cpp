//
// Created by yuanning on 18-6-28.
//

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

#include "lane_3d.h"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <algorithm>
//#include "xtensor/xarray.hpp"
//#include "xtensor/xio.hpp"

using namespace std;

class Lane3D {
public:
    Lane3D() {
        return;
    }

//    int fit(std::vector<cv::Point3d> &data, std::vector<cv::Point3d> &curve, int displayed_points=100) {
//        int ceil = 100;
//        if (data.size() > ceil) {
//            std::vector<unsigned int> indices(data.size());
//            iota(indices.begin(), indices.end(), 0);
//            random_shuffle(indices.begin(), indices.end());
//            std::vector<cv::Point3d> temp = std::vector<cv::Point3d>();
//            for (unsigned int i = 0; i < ceil; i++) {
//                temp.push_back(data[indices[i]]);
//            }
//            data = temp;
////            for (unsigned int i = 0; i < data.size(); i++) {
////                Point p = data[i];
////                cout << i << " " << p.x << " " << p.y << " " << p.z << endl;
////            }
//        }
//
//        //TODO: complete this method
//
//        return 0;
//    }
};

int getdir(std::string dir, std::vector<std::string> &files) {
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(std::string(dirp->d_name));
    }
    closedir(dp);
    sort(files.begin(), files.end());
    return 0;
}

int read_off(std::string path, std::vector<cv::Point3d> &lane_mask) {
    ifstream inFile;
    inFile.open(path);
    std::string title;
    inFile >> title;
//    cout << title << endl;
    int v, e, f;
    inFile >> v >> e >> f;
//    cout << v << " " << e << " " << f << endl;
    for (int i = 0; i < v; i++) {
        double x, y, z;
        inFile >> x >> y >> z;
        cv::Point3d p = cv::Point3d(x, y, z);
        lane_mask.push_back(p);
    }
    inFile.close();
    return 0;
}

int main(int argc, char *argv[]) {
    clock_t start;
    double duration;

    start = clock();

    std::string dir = "/home/yuanning/DeepMotion/pointclouds";
    std::vector<std::string> files = std::vector<std::string>();
    getdir(dir,files);

    for (unsigned int i = 2; i < files.size(); i++) {
//        cout << files[i] << endl;
        std::vector<cv::Point3d> lane_mask = std::vector<cv::Point3d>();
        read_off(dir + "/" + files[i], lane_mask);
//        for (unsigned int j = 0; j < lane_mask.size(); j++) {
//            Point p = lane_mask[j];
//            cout << p.x << " " << p.y << " " << p.z << endl;
//        }
//        std::vector<cv::Point3d> curve = std::vector<cv::Point3d>();
        Lane3D lane_3d = Lane3D();
//        lane_3d.fit(lane_mask, curve, 1000);
//        break;
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "Time used: " << duration << endl;
    return 0;
}
