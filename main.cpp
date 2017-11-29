#include <iostream>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include "siftmatch.h"
#include "time.h"
using namespace std;
using namespace cv;

#define CAMERA1 "camera1"
#define CAMERA2 "camera2"
#define MATCH "match"

int main() {
    SiftMatch siftMatch;
    IplImage *camera1 = nullptr;
    IplImage *camera2 = nullptr;

    CvCapture *capture1 = cvCreateCameraCapture(1);
    CvCapture *capture2 = cvCreateCameraCapture(3);

    cvSetCaptureProperty(capture1,CV_CAP_PROP_FRAME_WIDTH,1280);
    cvSetCaptureProperty(capture1,CV_CAP_PROP_FRAME_HEIGHT,720);

    cvSetCaptureProperty(capture2,CV_CAP_PROP_FRAME_WIDTH,1280);
    cvSetCaptureProperty(capture2,CV_CAP_PROP_FRAME_HEIGHT,720);

    clock_t startT, endT;

//    namedWindow(CAMERA1);
//    namedWindow(CAMERA2);
//    namedWindow(MATCH);
    int i = 0;

    while (true) {
        startT = clock();
        camera1 = cvQueryFrame(capture1);
        camera2 = cvQueryFrame(capture2);
        if (!camera1) {
            cerr<<"camera1 is not found"<<endl;
            return -1;
        }
        if (!camera2) {
            cerr<<"camera2 is not found"<<endl;
            return -1;
        }
//        siftMatch.input_img("pic/camera2/c2frame10.jpg", "pic/camera1/c1frame10.jpg");
        siftMatch.input_img(camera2, camera1);
        siftMatch.feature_detect();
        siftMatch.feature_match();
        siftMatch.mosaic();
        endT = clock();
        double totalT = (double)( (endT - startT)/(double)CLOCKS_PER_SEC );
        cout<<totalT<<endl;

        cvShowImage(MATCH, siftMatch.getRet());
//        cvSaveImage("pic/ret/ret_pic.jpg", siftMatch.getRet());

        // save image
//        if(i%10==0)
//        {
//            string name1 = "pic/camera1/c1frame";
//            string name2 = "pic/camera2/c2frame";
//            cvSaveImage((name1 + to_string(i) + ".jpg").c_str(), camera1);
//            cvSaveImage((name2 + to_string(i) + ".jpg").c_str(), camera2);
//        }
//        cvShowImage(CAMERA1, camera1);
//        cvShowImage(CAMERA2, camera2);
        cvWaitKey(10);
//        break;
//        if(i==200)
//            break;
//        i++;
    }

//    cvReleaseCapture(&capture1);
//    cvReleaseCapture(&capture2);
//    cvDestroyWindow(CAMERA1);
//    cvDestroyWindow(CAMERA2);

    return 0;
}