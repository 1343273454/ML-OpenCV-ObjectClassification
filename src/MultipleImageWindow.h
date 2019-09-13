/**
 * Mutliple Image Window
 *
 * Creates an Split Screen
 */

#ifndef MIW_h
#define MIW_h

#include <string>
#include <iostream>
// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class MultipleImageWindow
{
    public:
        /**
         *
         * @param string window_title 
         * @param int cols number of cols
         * @param int rows number of rows
         * @param int flags see highgui window documentation
         */
        MultipleImageWindow(const std::string& window_title, int cols, int rows, int flags= cv::WINDOW_AUTOSIZE);

        /**
         * Add new image to stack of window
         * @param Mat image
         * @param string title caption of image to show
         * @param bool render if wants to render the image
         * @return int position of image in stack
         */
        int addImage(const std::string& title, const cv::Mat& image, bool render= false);

        /**
         * Remove a image from position n
         */
        void removeImage(int pos);

        /**
         * Render/redraw/update window
         */
        void render();

    private:
        int m_cols;
        int m_rows;
        int m_canvas_width;
        int m_canvas_height;
        std::string m_window_title;
        std::vector<std::string> m_titles;
        std::vector<cv::Mat> m_images; 
        cv::Mat m_canvas;
};


#endif