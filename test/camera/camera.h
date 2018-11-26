#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using std::string;
using std::cout;
using std::endl;
using std::cerr;

string GetTegraPipeline(int width, int height, int fps) {
    // flip-method controls the flip direction of the image.
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" +
            std::to_string(width) + ", height=(int)" +
            std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
            "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

}


int Capture() {
    int width = 480;
    int height = 640;
    int fps = 120;

    string pipeline = GetTegraPipeline(width, height, fps);
    cout << "Using pipeline: " << pipeline << endl;

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Connection to camera failed" << endl;
        return EXIT_FAILURE;
    }

    cv::Mat frame;
    cv::Mat newFrame;
    while (true) {
        cap >> frame;
        //cv::resize(frame, newFrame, cv::Size(480, 720), 0, 0, cv::INTER_CUBIC);
        imshow("tx2 camera", frame);

        if ((char)cv::waitKey(10) == 'c') {
           cap.release();
           exit(EXIT_SUCCESS);
        }

    }
    return EXIT_SUCCESS;
}
