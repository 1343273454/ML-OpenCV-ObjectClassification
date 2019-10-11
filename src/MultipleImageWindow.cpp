#include "MultipleImageWindow.h"
#include <algorithm>

MultipleImageWindow::MultipleImageWindow(const std::string& window_title, int cols, int rows, int flags)
{
    this->m_window_title= window_title;
    this->m_cols= cols;
    this->m_rows= rows;
    cv::namedWindow(window_title, flags);
    
    this->m_canvas_width= 1200;
    this->m_canvas_height= 700;
    this->m_canvas= cv::Mat(this->m_canvas_height, this->m_canvas_width, CV_8UC3);     
    cv::imshow(this->m_window_title, this->m_canvas);
}

int MultipleImageWindow::addImage(const std::string& title, const cv::Mat& image, bool render)
{   
    std::vector<std::string>::iterator it= std::find(std::begin(m_titles),std::end(m_titles),title);
    if(it== m_titles.end()){
        this->m_titles.push_back(title);
        this->m_images.push_back(image);
    } else{
        this->m_images[it-std::begin(m_titles)]= image;
    }
    
    if(render)
        this->render();
    return this->m_images.size()-1;
}

void MultipleImageWindow::removeImage(int pos)
{
    this->m_titles.erase(this->m_titles.begin()+pos);
    this->m_images.erase(this->m_images.begin()+pos);
}

void MultipleImageWindow::render()
{
    // Clean our canvas
    this->m_canvas.setTo( cv::Scalar(20,20,20) );
    // width and height of cell. add 10 px of padding between images
    int cell_width= (m_canvas_width/m_cols);
    int cell_height= (m_canvas_height/m_rows);
    
    int max_images=(this->m_images.size()>m_cols*m_rows)?m_cols*m_rows:this->m_images.size();
    int i=0;
    std::vector<std::string>::iterator titles_it= this->m_titles.begin();
    for(std::vector<cv::Mat>::iterator it= this->m_images.begin(); it!= this->m_images.end(); ++it)
    {
        std::string title= *titles_it;
        int cell_x= (cell_width)*((i)%m_cols);
        int cell_y= (cell_height)*floor((i)/(float)m_cols);
        cv::Rect mask(cell_x, cell_y, cell_width, cell_height);
        
        //For each cell draw an image if exists
        cv::Mat cell(this->m_canvas, mask);
        // resize image to cell size
        cv::Mat resized;
        double cell_aspect= (double)cell_width/(double)cell_height;
        cv::Mat img= *it;
        double img_aspect= (double)img.cols/(double)img.rows;
        double f=(cell_aspect<img_aspect)?(double)cell_width/(double)img.cols:(double)cell_height/(double)img.rows;
        resize(img, resized, cv::Size(0,0), f, f);
        if(resized.channels()==1){
            cvtColor(resized, resized, cv::COLOR_GRAY2BGR); 
        }

        // Assign the image
        cv::Mat sub_cell(this->m_canvas, cv::Rect(cell_x+((cell_width-resized.cols)/2),cell_y+((cell_height-resized.rows)/2),resized.cols, resized.rows));

        resized.copyTo(sub_cell);

        putText(cell, title.c_str(),cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200,0,0), 1, cv::LINE_AA);

        // Draw a rectangle for each cell mat 
        cv::rectangle(m_canvas, cv::Rect(cell_x, cell_y, cell_width, cell_height), cv::Scalar(200,200,200), 1);

        i++;
        ++titles_it;
        if(i==max_images)
            break;
    }

    // show image
    cv::imshow(this->m_window_title, this->m_canvas);
    // save image
    cv::imwrite("../img/Output.jpg",this->m_canvas);
}
