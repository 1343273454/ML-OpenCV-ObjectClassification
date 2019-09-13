#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>
#include <vector>

// OpenCV includes
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "MultipleImageWindow.h"
#include "PreProcessImage.hpp"

/**
 *  lightMethod:   0: Difference operation is applied
 *                 1: The division operation is applied
 * 
 *   segMethod:    1: the connected components method for segment is applied
 *                 2: the connected components method with the statistics area is applied
 *                 3: the find contours method is applied for Segmentation
 */
const char *keys =
    {
        "{help h usage ? | | print this message}"
        "{@image || Image to process}"
        "{@lightPattern || Image light pattern to apply to image input}"
        "{lightMethod | 1 | Method to remove background light, 0 difference, 1 div, 2 no light removal }"
        "{segMethod | 1 | Method to segment: 1 connected Components, 2 connected components with stats, 3 find Contours }"};

std::shared_ptr<MultipleImageWindow> miw;
// support vector machine model
cv::Ptr<cv::ml::SVM>svm;
cv::Scalar green(0,255,0), blue (255,0,0), red (0,0,255);
cv::Mat light_pattern;


//The output of a function is a vector of vectors of floats. In other words, it is a 
//matrix where each row contains the features of each object that's detected.
// left and top vectors are to store left-top position of each object and put a label later
std::vector<std::vector<float>> ExtractFeatures(cv::Mat img, std::vector<int> *left = NULL, std::vector<int> *top = NULL)
{
    std::vector<std::vector<float>> output;
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat input = img.clone();

    std::vector<cv::Vec4i> hierarchy;
    // find contours in a binary image, contours are useful for shape analisis
    cv::findContours(input, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    // Check the number of objects detected
    if (contours.size() == 0)
    {
        return output;
    }
    //random number generator
    cv::RNG rng(0xFFFFFFFF);
    for (auto i = 0; i < contours.size(); i++)
    {

        cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        //draw the object in white on a black background 
        cv::drawContours(mask, contours, i, cv::Scalar(1), cv::FILLED, cv::LINE_8, hierarchy, 1);
        // calculate area of shape
        cv::Scalar area_s = cv::sum(mask);
        float area = area_s[0];
        //All objects with an area less than the minimum threshold area that we considered will be discarded
        if (area > 500)
        { //if the area is greater than min.

            //calculate aspect ratio,Finds a rotated rectangle of the minimum area enclosing the input 2D point set 
            cv::RotatedRect r = minAreaRect(contours[i]);
            float width = r.size.width;
            float height = r.size.height;
            float ar = (width < height) ? height / width : width / height;

            // add features
            std::vector<float> row={area,ar};
            output.push_back(row);
            if (left != NULL)
            {
                left->push_back((int)r.center.x);
            }
            if (top != NULL)
            {
                top->push_back((int)r.center.y);
            }

            // Add image to the multiple image window class
            miw->addImage("Extract Features", mask * 255);
            miw->render();
            cv::waitKey(10);
        }
    }
    return output;
}

void plotTrainData(cv::Mat trainData, cv::Mat labels, float *error = NULL)
{
    float area_max, ar_max, area_min, ar_min;
    area_max = ar_max = 0;
    area_min = ar_min = 99999999;

    
    // Get the min and max of each feature for normalize plot image
    for (int i = 0; i < trainData.rows; i++)
    {
        float area = trainData.at<float>(i, 0);
        float ar = trainData.at<float>(i, 1);
        if (area > area_max)
            area_max = area;
        if (ar > ar_max)
            ar_max = ar;
        if (area < area_min)
            area_min = area;
        if (ar < ar_min)
            ar_min = ar;
    }

    // Create Image for plot
    cv::Mat plot = cv::Mat::zeros(512, 512, CV_8UC3);
    // Plot each of two features in a 2D graph using an image
    // where x is area and y is aspect ratio
    for (int i = 0; i < trainData.rows; i++)
    {
        // Set the X y pos for each data
        float area = trainData.at<float>(i, 0);
        float ar = trainData.at<float>(i, 1);
        int x = (int)(512.0f * ((area - area_min) / (area_max - area_min)));
        int y = (int)(512.0f * ((ar - ar_min) / (ar_max - ar_min)));

        // Get label
        int label = labels.at<int>(i);
        // Set color depend of label
        cv::Scalar color;
        if (label == 0)
            color = green; // NUT
        else if (label == 1)
            color = blue; // RING
        else if (label == 2)
            color = red; // SCREW

        cv::circle(plot, cv::Point(x, y), 3, color, -1, 8);
    }

    if (error != NULL)
    {
        std::stringstream ss;
        ss << "Error: " << *error << "\%";
        putText(plot, ss.str().c_str(), cv::Point(20, 512 - 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }
    miw->addImage("Plot", plot);
}

/**
* Read all images in a folder creating the train and test vectors
* @param folder string name
* @param label assigned to train and test data
* @param number of images used for test and evaluate algorithm error
* @param trainingData vector where store all features for training
* @param reponsesData vector where store all labels corresopinding for training data, in this case the label values
* @param testData vector where store all features for test, this vector as the num_for_test size
* @param testResponsesData vector where store all labels corresponiding for test, has the num_for_test size with label values
* @return true if can read the folder images, false in error case
**/
bool readFolderAndExtractFeatures(std::string folder, int label, int num_for_test,
                                std::vector<float> &trainingData, std::vector<int> &responsesData,
                                std::vector<float> &testData, std::vector<float> &testResponsesData)
{
    cv::VideoCapture images;
    if (images.open(folder) == false)
    {
        std::cout << "Can not open the folder images" << std::endl;
        return false;
    }
    cv::Mat frame;
    int img_index = 0;
    while (images.read(frame))
    {
        PreProcessImg p(frame, light_pattern);
        p.StartPreProcess();
        // Extract features
        std::vector<std::vector<float>> features = ExtractFeatures(p.getBinaryImg());
        for (int i = 0; i < features.size(); i++)
        {
            if (img_index >= num_for_test)
            {
                trainingData.push_back(features[i][0]);
                trainingData.push_back(features[i][1]);
                responsesData.push_back(label);
            }
            else
            {
                testData.push_back(features[i][0]);
                testData.push_back(features[i][1]);
                testResponsesData.push_back((float)label);
            }
        }
        img_index++;
    }
    return true;
}

void trainAndTest()
{
    std::vector<float> trainingData;
    std::vector<int> responsesData;
    std::vector<float> testData;
    std::vector<float> testResponsesData;

    int num_for_test = 20;

    // Get the nut images
    readFolderAndExtractFeatures("../img/nut/tuerca_%04d.pgm", 0, num_for_test, trainingData, responsesData, testData, testResponsesData);
    // Get and process the ring images
    readFolderAndExtractFeatures("../img/ring/arandela_%04d.pgm", 1, num_for_test, trainingData, responsesData, testData, testResponsesData);
    // get and process the screw images
    readFolderAndExtractFeatures("../img/screw/tornillo_%04d.pgm", 2, num_for_test, trainingData, responsesData, testData, testResponsesData);

    std::cout << "Num of train samples: " << responsesData.size() << std::endl;

    std::cout << "Num of test samples: " << testResponsesData.size() << std::endl;

    // Merge all data  
    // From Vector to Mat.  transform the training data(features) in a big matrix with many rows 
    // and two columns(the number of features)
    cv::Mat trainingDataMat(trainingData.size() / 2, 2, CV_32FC1, &trainingData[0]);
    cv::Mat responses(responsesData.size(), 1, CV_32SC1, &responsesData[0]);

    cv::Mat testDataMat(testData.size() / 2, 2, CV_32FC1, &testData[0]);
    cv::Mat testResponses(testResponsesData.size(), 1, CV_32FC1, &testResponsesData[0]);

    cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, responses);

    // Set up SVM's parameters 
    // First, we are going to set up the basic model parameters
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setNu(0.05);
    svm->setKernel(cv::ml::SVM::CHI2);
    svm->setDegree(1.0);
    svm->setGamma(2.0);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    //we are going to create the model by calling the train method and using trainingDataMat and response matrices as a TrainData
    svm->train(tdata);
    
    if (testResponsesData.size() > 0)
    {
        std::cout << "Evaluation" << std::endl;
        std::cout << "==========" << std::endl;
        // Test the ML Model
        cv::Mat testPredict;
        //The predict function makes it possible to make multiple predictions at the same time, 
        //giving a matrix as the result instead of only one row or vector
        svm->predict(testDataMat, testPredict);
        std::cout << "Prediction Done" << std::endl;
        // Error calculation
        cv::Mat errorMat = testPredict != testResponses;
        float error = 100.0f * countNonZero(errorMat) / testResponsesData.size();
        std::cout << "Error: " << error << "%" << std::endl;
        // Plot training data with error label
        plotTrainData(trainingDataMat, responses, &error);
    }
    else
    {
        plotTrainData(trainingDataMat, responses);
    }
}

/**
 * This basic function applies a blur to an input image by using a big kernel size relative to the image size. 
 * From the code, it is one-third of the original width and height.
 * Estimate the background image in case with don't have a Light pattern
 */
cv::Mat calculateLightPattern(cv::Mat img)
{
    cv::Mat pattern;
    // Basic and effective way to calculate the light pattern from one image
    cv::blur(img, pattern, cv::Size(img.cols / 3, img.cols / 3));
    return pattern;
}

bool processInput(const cv::CommandLineParser &p, cv::Mat &out_img,
                  int &out_method_light, int &out_method_seg)
{

    // if requires hel show
    if (p.has("help"))
    {
        p.printMessage();
        return false;
    }

    // Check if params are correctly parsed in his variables
    if (!p.check())
    {
        p.printErrors();
        return false;
    }

    std::string img_file = p.get<std::string>(0);
    std::string light_pattern_file = p.get<std::string>(1);
    out_method_light = p.get<int>("lightMethod");
    out_method_seg = p.get<int>("segMethod");
    //convert image to the single channel grayscale image
    out_img = cv::imread(img_file,0);
    if (out_img.data == NULL)
    {
        std::cout << "Error Loading the image\n";
        return false;
    }

    light_pattern = cv::imread(light_pattern_file, 0);
    if (light_pattern.data == NULL)
    {
        //Calculate light pattern
        light_pattern = calculateLightPattern(out_img);
    }
    cv::medianBlur(light_pattern,light_pattern, 3);
    return true;
}

void testTrainedModel(const std::unique_ptr<PreProcessImg>& p,cv::Mat & img_out){
     // Extract features
    std::vector<int> pos_top, pos_left;
    std::vector<std::vector<float>> features = ExtractFeatures(p->getBinaryImg(), &pos_left, &pos_top);

    std::cout << "Num objects extracted features " << features.size() << std::endl;

    for (int i = 0; i < features.size(); i++)
    {

        std::cout << "Data Area AR: " << features[i][0] << " " << features[i][1] << std::endl;

        cv::Mat trainingDataMat(1, 2, CV_32FC1, &features[i][0]);
        std::cout << "Features to predict: " << trainingDataMat << std::endl;
        float result = svm->predict(trainingDataMat);
        std::cout << result << std::endl;

        std::stringstream ss;
        cv::Scalar color;
        if (result == 0)
        {
            color = green; // NUT
            ss << "NUT";
        }
        else if (result == 1)
        {
            color = blue; // RING
            ss << "RING";
        }
        else if (result == 2)
        {
            color = red; // SCREW
            ss << "SCREW";
        }

        putText(img_out,
                ss.str(),
                cv::Point2d(pos_left[i], pos_top[i]),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                color);
    }
}

int main(int argc, const char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Object Classification");
    
    cv::Mat img;
    int method_light;
    int method_seg;
    if (!processInput(parser, img, method_light, method_seg))
    {
        return -1;
    }
    std::unique_ptr<PreProcessImg> p;
    p = std::make_unique<PreProcessImg>(img, light_pattern, method_light, method_seg);
    p->StartPreProcess();

	miw= std::make_shared<MultipleImageWindow>("Main window", 2, 2, cv::WINDOW_AUTOSIZE);
    trainAndTest();

    cv::Mat img_output= img.clone();
    cv::cvtColor(img_output,img_output,cv::COLOR_GRAY2BGR);
    testTrainedModel(p,img_output);

    // Show images
    miw->addImage("Binary image", p->getBinaryImg());
    miw->addImage("Result", img_output);
    miw->render();

    cv::waitKey(0);

    return 0;
}