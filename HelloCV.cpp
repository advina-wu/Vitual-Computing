#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/imgcodecs/imgcodecs.hpp>

#define PI 3.14159265

// default threshold flag is set to Binary
using namespace cv;
std::string input_image_path, output_filename;
Mat input_image, grey, canny, hough_lines, long_lines, no_vertical_lines, horizon;

//Binary threshold variable
int threshold = 70;

//Polynomial regression function
vector<double> fitPoly(vector<Point> points, int n)
{
  //Number of points
  int nPoints = points.size();

  //Vectors for all the points' xs and ys
  vector<float> xValues = vector<float>();
  vector<float> yValues = vector<float>();

  //Split the points into two vectors for x and y values
  for(int i = 0; i < nPoints; i++)
  {
    xValues.push_back(points[i].x);
    yValues.push_back(points[i].y);
  }

  //Augmented matrix
  double matrixSystem[n+1][n+2];
  for(int row = 0; row < n+1; row++)
  {
    for(int col = 0; col < n+1; col++)
    {
      matrixSystem[row][col] = 0;
      for(int i = 0; i < nPoints; i++)
        matrixSystem[row][col] += pow(xValues[i], row + col);
    }

    matrixSystem[row][n+1] = 0;
    for(int i = 0; i < nPoints; i++)
      matrixSystem[row][n+1] += pow(xValues[i], row) * yValues[i];

  }

  //Array that holds all the coefficients
  double coeffVec[n+2] = {};  // the "= {}" is needed in visual studio, but not in Linux

  //Gauss reduction
  for(int i = 0; i <= n-1; i++){
    for (int k=i+1; k <= n; k++)
    {
      double t=matrixSystem[k][i]/matrixSystem[i][i];

      for (int j=0;j<=n+1;j++)
        matrixSystem[k][j]=matrixSystem[k][j]-t*matrixSystem[i][j];

    }
  }

  //Back-substitution
  for (int i=n;i>=0;i--)
  {
    coeffVec[i]=matrixSystem[i][n+1];
    for (int j=0;j<=n+1;j++)
      if (j!=i)
        coeffVec[i]=coeffVec[i]-matrixSystem[i][j]*coeffVec[j];

    coeffVec[i]=coeffVec[i]/matrixSystem[i][i];
  }

  //Construct the cv vector and return it
  vector<double> result = vector<double>();
  for(int i = 0; i < n+1; i++)
    result.push_back(coeffVec[i]);
  return result;
}

//Returns the point for the equation determined
//by a vector of coefficents, at a certain x location
Point pointAtX(vector<double> coeff, double x)
{
  double y = 0;
  for(int i = 0; i < coeff.size(); i++)
  y += pow(x, i) * coeff[i];
  return Point(x, y);
}


int main( int argc, char *argv[])
{
  printf ("OpenCV version: %d.%d \n ", CV_MAJOR_VERSION, CV_MINOR_VERSION);

  if (argc < 2){
    std::cout << "Please provide the path to the input image " << std::endl;
    return 0;
  }


  input_image_path = argv[1];

  input_image = imread(input_image_path, IMREAD_GRAYSCALE);
  // input_image = imread(input_image_path, IMREAD_UNCHANGED);
  hough_lines = imread(input_image_path, IMREAD_UNCHANGED);
  long_lines  = imread(input_image_path, IMREAD_UNCHANGED);
  no_vertical_lines = imread(input_image_path, IMREAD_UNCHANGED);
  horizon = imread(input_image_path, IMREAD_UNCHANGED);

  if(input_image.empty()){
    std::cout << "Failed to read image: " << argv[1] << std::endl;
    return 0;
  }

  // Copy edges to the images that will display the results in BGR
  cvtColor(input_image, grey, COLOR_GRAY2BGR, 0);
  // Edge detection by applying canny filter
  // (source, output, lower threshold, upper threshold, aparture, L2Gradient)
  Canny(grey, canny, 100, 150, 3);

// this willl store all lines
  vector<Vec4i> linesP; // will hold the results of the detection
  // ProbabilisticHough tranform#include <stdio.h>
  // (output of canny, lines vector,
  // rho - resolution of the parameter r in a pizels, t
  // theta, resolution in radians CV_PI/180 = 1 degree, (angle resoltion)
  // threshold, minLineLength/MaxLineGap default = 0)
  HoughLinesP(canny, linesP, 1, CV_PI/180, 50, 0, 3); // runs the actual detection

  int max = 0;
  int min = -1;
  int total_len = 0;

  for( size_t i = 0; i < linesP.size(); i++ ){
    Vec4i l = linesP[i];

    int len = sqrt( pow((l[2] - l[0]), 2) + pow((l[3] - l[1]), 2));

    // to get the max, min and total length.
    if (len > max){
      max = len;
    }
    else if (len < min){
      min = len;
    }

    total_len += len;

    line( hough_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, 16);

  }

  int difference = max - min;
  int avg_len = total_len/linesP.size();

  vector<Vec4i> noShortlinesP;
  for( size_t i = 0; i < linesP.size(); i++ )
  {
      Vec4i l = linesP[i];

      //  to find the length
      int len = sqrt( pow((l[2] - l[0]), 2) + pow((l[3] - l[1]), 2));

      // use different threshold values depending on the length
      if (difference >= 130){
        if (len >= avg_len + 110){
          // Draw the lines
          line( long_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, 16);
          noShortlinesP.push_back(l);
        }
      }
      else{
        if (len >= avg_len + 55){
          // Draw the lines
          line( long_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, 16);
          noShortlinesP.push_back(l);
        }
      }
  }

  // lines' length
  vector<Vec4i> noVerticalLine;
  vector<Point> points;

  for( size_t i = 0; i < noShortlinesP.size(); i++ )
  {
      Vec4i l = noShortlinesP[i];

      // find the angle of the line
      double ang = atan2((l[3] - l[1]), (l[2] - l[0])) * 180 / PI;
      std::cout << "printing ang " << ang << std::endl;

      if ((ang < 30 && -30 < ang) | (ang < -150 && 150 < ang)){
        std::cout << "am in" << std::endl;

        line(no_vertical_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, 16);
        noVerticalLine.push_back(l);
        // push x and y into points thats used for polynomial regression
        points.push_back(Point(l[0], l[1]));
        points.push_back(Point(l[2], l[3]));
      }
  }

  // polynomial regression
  vector<double> coe = fitPoly(points, 2);

  for ( size_t i = 0; i < 5000; i ++ ){
    Point p1 = pointAtX(coe, i);
    Point p2 = pointAtX(coe, i+1);

    line(horizon, p1, p2, Scalar(0,0,255), 1, 16);
  }


  // Show results
  imshow("Canny Image", canny);
  imshow("Probabilistic Hough Lines", hough_lines);
  imshow("Short Lines Removed", long_lines);
  imshow("Horizontal Lines", no_vertical_lines);
  imshow("Horizon", horizon);
  waitKey(0);
  // it saves the image with the threshold value just before closing the window.
  imwrite("Canny_Image.jpg", canny);
  imwrite("Probabilistic_Hough_Lines.jpg", hough_lines);
  imwrite("Short_lines_Removed.jpg", long_lines);
  imwrite("Horizontal_Lines.jpg", no_vertical_lines);
  imwrite("Horizon.jpg", horizon);



  destroyAllWindows();
  return 0;

}
