#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

static void help() {
  cout << "\n-----------------------------------------------------------------"
          "-"
          "------------------\n"
       << " This program shows the multiview reconstruction capabilities in "
          "the \n"
       << " OpenCV Structure From Motion (SFM) module.\n"
       << " It reconstruct a scene from a set of 2D images \n"
       << " Usage:\n"
       << "        example_sfm_scene_reconstruction <path_to_file> <f> <cx> "
          "<cy>\n"
       << " where: path_to_file is the file absolute path into your system "
          "which contains\n"
       << "        the list of images to use for reconstruction. \n"
       << "        f  is the focal length in pixels. \n"
       << "        cx is the image principal point x coordinates in pixels. \n"
       << "        cy is the image principal point y coordinates in pixels. \n"
       << "-------------------------------------------------------------------"
          "-"
          "----------------\n\n"
       << endl;
}

static int getdir(const string _filename, vector<String> &files) {
  ifstream myfile(_filename.c_str());
  if (!myfile.is_open()) {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);
  } else {
    ;
    size_t found = _filename.find_last_of("/\\");
    string line_str, path_to_file = _filename.substr(0, found);
    while (getline(myfile, line_str)) files.push_back(path_to_file + string("/") + line_str);
  }
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    help();
    exit(0);
  }

  vector<String> images_paths;
  getdir(argv[1], images_paths);

  for (auto it : images_paths) std::cout << it << std::endl;

  float f = atof(argv[2]), cx = atof(argv[3]), cy = atof(argv[4]);
  Matx33d K = Matx33d(f, 0, cx, 0, f, cy, 0, 0, 1);

  bool is_projective = true;
  vector<Mat> Rs_est, ts_est, points3d_estimated;
  reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);
}