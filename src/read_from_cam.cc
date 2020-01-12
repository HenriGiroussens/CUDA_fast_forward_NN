#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;

int main()
{
    //-- 1. Load the cascades
    if( !face_cascade.load( "../haarcascades/haarcascade_frontalface_default.xml" ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    }
    int camera_device = 0;
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}

void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 5, 0, Size(48, 48) );
    for (auto & face : faces)
    {
        Point X(face.x, face.y);
        Point Y(face.x + face.width, face.y + face.height);
        rectangle(frame, X, Y, Scalar( 255, 0, 255 ));
        Mat faceROI;
        resize(frame_gray(face), faceROI, Size(48, 48));
        faceROI = faceROI.reshape(1, 48*48);
        double* face_arr = static_cast<double *>(malloc(48 * 48 * sizeof(double)));
        for (int i = 0; i < 48*48; ++i) {
            face_arr[i] = (double)(faceROI.data[i]);
        }

        // CALL NN HERE
    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}