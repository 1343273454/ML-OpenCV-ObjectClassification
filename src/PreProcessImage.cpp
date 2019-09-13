#include "PreProcessImage.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

cv::Scalar _randomColor(cv::RNG &rng)
{
    unsigned icolor = (unsigned)rng;
    return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void PreProcessImg::removeNoise(){
    if (m_original_img.channels() == 3)
        cv::cvtColor(m_original_img, m_original_img, cv::COLOR_RGB2GRAY);
    // Smoothes an image using median filter with 3 x 3 aperture.
    cv::medianBlur(m_original_img, m_img_no_noise, 3);
}

void PreProcessImg::removeLight(){
    if(m_light_method!=2){
        // if method is normalization
    if (m_light_method == 1)
    {
        // Require change our image to 32 float for division
        cv::Mat img32, pattern32;
        // CV_8U -> CV_32F
        m_img_no_noise.convertTo(img32, CV_32F);
        m_light_pattern.convertTo(pattern32, CV_32F);

        // Divide the image by the pattern
        m_img_no_light = 1 - (img32 / pattern32);
        // convert 8 bit format and scale
        m_img_no_light.convertTo(m_img_no_light, CV_8U, 255);
    }
    else
    {
        m_img_no_light = m_light_pattern - m_img_no_noise;
    }
    }
}

void PreProcessImg::binarize(){
    if (m_light_method != 2)
    {
        //The function applies fixed-level thresholding to a multiple-channel 
        //array. The function is typically used to get a bi-level (binary) 
        //image out of a grayscale image 
        cv::threshold(m_img_no_light, m_img_threshold, 30, 255, cv::THRESH_BINARY);
    }
    else
    {
        cv::threshold(m_img_no_light, m_img_threshold, 140, 255, cv::THRESH_BINARY_INV);
    }
}

void PreProcessImg::StartPreProcess(){
    removeNoise();
    removeLight();
    binarize();
}