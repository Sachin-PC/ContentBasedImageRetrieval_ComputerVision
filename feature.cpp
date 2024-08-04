/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Containg logic to calculate feature vectors, distance and ranks
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "math.h"

using namespace cv;
using namespace std;

// Defining the typedef of methods used as function parameters in this file
typedef int (*FeatureVectorFunction)(const cv::Mat&, std::vector<float>&, int);
typedef double (*DistanceScoreFunction)(std::vector<float>, std::vector<float>);
typedef double (*RankScoreFunction)(std::vector<float>);


// Defining the prototypes of methods used in this file
int append_image_data_csv( char *filename, char *image_filename, std::vector<float> &image_data, int reset_file );

/**
 * @brief calculates Sum of squared difference as distance metric 
 * @param[in] targetImageVector The feature vector of target image.
 * @param[in] imageFilterVector TThe feature vector of considered image.
 * @return 0.
*/
double getSumSquaredDifference(vector<float>targetImageVector, std::vector<float> imageFilterVector){
    int n = targetImageVector.size();
    double distanceValue = 0;
    for(int i=0;i<n;i++){
        distanceValue += (targetImageVector[i] - imageFilterVector[i])*(targetImageVector[i] - imageFilterVector[i]);
    }
    return distanceValue;

}

/**
 * @brief sorts the distances vector and also keep the indexs mapped. 
 * @param[in] distanceVector The distance vector to be sorted.
 * @param[in] sortedDistances The sorted distance vector.
 * @return 0.
*/
int sortDistances(std::vector<double> distanceVector, vector<pair<double,int> > &sortedDistances){
    int n = distanceVector.size();
    for(int i=0;i<n;i++){
        sortedDistances.push_back(make_pair(distanceVector[i], i));
    }
    sort(sortedDistances.begin(), sortedDistances.end());

    return 0;
}

/**
 * @brief calculates histogram intersection as distance metric 
 * @param[in] targetImageVector The feature vector of target image.
 * @param[in] imageFilterVector TThe feature vector of considered image.
 * @return 0.
*/
double gethistogramIntersection(vector<float>targetImageVector, std::vector<float> imageFilterVector){
    int n = targetImageVector.size();
    double distanceValue = 0;
    for(int i=0;i<n;i++){
        distanceValue += min(targetImageVector[i],imageFilterVector[i]);
    }
    return distanceValue;
}

/**
 * @brief calculates Sthe sunset score of the given image
 * @param[in] imageFilterVector TThe feature vector of considered image.
 * @return 0.
*/
double getSunsetScores(std::vector<float> imageFilterVector){
    if(imageFilterVector.size() != 1){
        cout<<"Not Possibl;e. Sunset score should be only 1 value. Sunsetscore size = "<<imageFilterVector.size()<<endl;
        exit(0);
    }
    return imageFilterVector[0];
}

/**
 * @brief calculates the sunset score of the given image interms of intensity.
 * @param[in] imageFilterVector TThe feature vector of considered image.
 * @return 0.
*/
double getSunsetIntensityScore(std::vector<float> imageFilterVector){

    int scoreVectorSize, max_index=0, j=0;
    double max=0, diffVal=0;
    scoreVectorSize = imageFilterVector.size();
    for(j=scoreVectorSize/2;j< scoreVectorSize; j++){
        if(imageFilterVector[j] > max){
            max = imageFilterVector[j];
            max_index = j;
        }
    }
    j = max_index -1;
    int factor = 3;
    int indexCount=3;
    int prevScore = max;
    while(indexCount > 0){
        if(j < scoreVectorSize/2 || imageFilterVector[j] > prevScore){
            break;
        }
        diffVal += factor*(max - imageFilterVector[j]);
        j--;
        indexCount--;
    }
    j = max_index + 1;
    indexCount=3;
    factor = 3;
    prevScore = max;
    while(indexCount > 0){
        if(j >= scoreVectorSize || imageFilterVector[j] > prevScore){
            break;
        }
        diffVal += factor*(max - imageFilterVector[j]);
        j++;
        indexCount--;
    }
    // diffVal = diffVal/600;
    // sunsetScore = imageFilterVector[0];
    double sunsetIntensityScore = diffVal/600;
    return sunsetIntensityScore;
}

/**
 * @brief calculates the sunset score of the given image interms of intensity Histogram.
 * @param[in] imageFilterVector TThe feature vector of considered image.
 * @return 0.
*/
double getSunsetIntensityHistogramScore(std::vector<float> imageFilterVector){

    int scoreVectorSize, max_index=0, j=0;
    double max=0, diffVal=0;
    scoreVectorSize = imageFilterVector.size();
    for(j=1;j< scoreVectorSize; j++){
        if(imageFilterVector[j] > max){
            max = imageFilterVector[j];
            max_index = j;
        }
    }
    j = max_index -1;
    int factor = 3;
    int indexCount=3;
    int prevScore = max;
    while(indexCount > 0){
        if(j < 1 || imageFilterVector[j] > prevScore){
            break;
        }
        diffVal += factor*(max - imageFilterVector[j]);
        j--;
        indexCount--;
    }
    j = max_index + 1;
    indexCount=3;
    factor = 3;
    prevScore = max;
    while(indexCount > 0){
        if(j >= scoreVectorSize || imageFilterVector[j] > prevScore){
            break;
        }
        diffVal += factor*(max - imageFilterVector[j]);
        j++;
        indexCount--;
    }
    // diffVal = diffVal/600;
    // sunsetScore = imageFilterVector[0];
    double sunsetIntensityScore = diffVal/600;
    return sunsetIntensityScore;
}

/**
 * @brief Gets the xSobel/ySobel image of the src image.
 * This function takes input a image and finds the 
 * xSobel/ySobel filtered image bsed on kernel passed.
 * @param[in] src The input image.
 * @param[in] dst The output image which should contain the result image.
 * @param[in] sobelKernel The filter kernel used as seperable filter.
 * @param[in] kernel_multiplier used with kernel filter to be multiplied with piexl value.
 * @return 0.
 */
int sobel3x3SeperableFilter(const cv::Mat &src, cv::Mat &dst, int sobelKernel[3], int kernel_multiplier[3]){
    dst.create(src.size(), CV_16SC3);

    //checks if the input image contains data
    if(src.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = src.rows;
    int numCols = src.cols;
    int numChannels = src.channels();

    int p=0,q=0,kernel_i=0,kernel_j=0, filter_value=0, filters_sum=0, colour_channel=0;
    int pixelColourValue=0;

    //apply the kernel given and update the pixel values
    for(int i=1;i<numRows-1;i++){
        for(int j=1;j<numCols-1;j++){
            for(colour_channel=0;colour_channel<3;colour_channel++){
                pixelColourValue = static_cast<int>(src.at<Vec3b>(i, j)[colour_channel]);
                kernel_j = j-1;
                filter_value = 0;
                p = i-1;
                for(kernel_i=0;kernel_i<3;kernel_i++){
                    q = j-1;
                    for(kernel_j=0;kernel_j<3;kernel_j++){
                        filter_value += sobelKernel[kernel_j]*kernel_multiplier[kernel_i]*src.at<Vec3b>(p, q)[colour_channel];
                        q++;
                    }
                    p++;
                }
                dst.at<Vec3s>(i, j)[colour_channel] = filter_value;
            }
        }
    }
    return 0;
}


/**
 * @brief Gets the Sobel magnitude image of the src image.
 * This function takes input a image and finds the 
 * sobel magnitude filtered image.
 * @param[in] sx The SobelX filter image.
 * @param[in] sy The SobelY filter image.
 * @param[in] dst The output image which should contain the result image.
 * @return 0.
*/
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){

    dst.create(sx.size(), CV_16SC3);

    //checks if the input image contains data
    if(sx.data == NULL || sy.data == NULL) {
        printf("Unable to read image ");
        exit(-1);
    }

    int numRows = sx.rows;
    int numCols = sx.cols;
    int numChannels = sx.channels();

    int sx_value=0, sy_value = 0, mag_value=0;

    //calculate the eulidian distance and update the pixel values
    for(int i=0;i<numRows-1;i++){
        for(int j=0;j<numCols;j++){
            for(int k =0; k< numChannels; k++){
                sx_value = sx.at<Vec3s>(i, j)[k];
                sy_value = sy.at<Vec3s>(i, j)[k];
                mag_value = sqrt(sx_value*sx_value + sy_value*sy_value);
                dst.at<Vec3s>(i, j)[k] = mag_value;
            }
        }
    }
    return 0;
}


/**
 * @brief Gets the Sobel magnitude image of the src image.
 * This function takes input a image and finds the 
 * sobel magnitude filtered image.
 * @param[in] src The input image.
 * @param[in] dst The SobelM output image.
 * @return 0.
*/
int getMSobel(const Mat &src, Mat &dst){

    Mat sobelx,sobely;
    int sobelXKernel[3] = {-1, 0, 1};
    int kernelx_multiplier[3] = {1,2,1};
    sobel3x3SeperableFilter(src, sobelx, sobelXKernel, kernelx_multiplier);

    int sobelYKernel[3] = {1, 2, 1};
    int kernely_multiplier[3] = {1,0,-1};
    sobel3x3SeperableFilter(src, sobely, sobelYKernel, kernely_multiplier);

    magnitude( sobelx, sobely, dst );
    
    return 0;
}

/**
 * @brief calculates baseline matching feature vector for given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the feature results of the given imagee.
 * @param[in] numBins for this case, this value is not considered
 * @return 0.
*/
int getBaselineMatchinFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){
    int numRows, numCols, numChannels, centerRow, centerColumn, start_index_row,start_index_col;
    float pixelValue;
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();

    centerRow = numRows/2;
    centerColumn = numCols/2;
    start_index_row = centerRow - 3;
    start_index_col = centerColumn - 3;
    for(int i=start_index_row;i<start_index_row+7;i++){
        for(int j=start_index_col;j<start_index_col+7;j++){
            for(int channel =0; channel<3;channel++){
                pixelValue = src.at<Vec3b>(i, j)[channel];
                featureVector.push_back(pixelValue);
            }
        }
    }

    return 0;
}

/**
 * @brief calculates rg chromaticity feature vector for given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the rg chromaticityfeature results of the given imagee.
 * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int getRGChromaticityFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){
    int numRows, numCols, numChannels;
    int blueVal, redVal, greenVal, r_index, g_index;
    float r_chromaticity,g_chromaticity;
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();
    int rgChromaticity[numBins][numBins];
    for(int i=0;i<numBins;i++){
        for(int j=0;j<numBins;j++){
            rgChromaticity[i][j] = 0;
        }
    }
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            blueVal = src.at<Vec3b>(i, j)[0];
            greenVal = src.at<Vec3b>(i, j)[1];
            redVal = src.at<Vec3b>(i, j)[2];
            r_chromaticity = (float)redVal/(blueVal+greenVal+redVal);
            g_chromaticity = (float)greenVal/(blueVal+greenVal+redVal);
            if(r_chromaticity <0 || r_chromaticity > 1){
                cout<<"Not possible. r_chromaticity value = "<<r_chromaticity<<endl;
            }
            if(g_chromaticity <0 || g_chromaticity > 1){
                cout<<"Not possible. g_chromaticity value = "<<g_chromaticity<<endl;
            }
            r_index = (int)(r_chromaticity*(numBins-1)+0.5);
            g_index = (int)(g_chromaticity*(numBins-1)+0.5);
            if(r_index <0 || r_index > numBins-1){
                cout<<"Not possible. r_index value = "<<r_index<<endl;
            }
            if(g_index <0 || g_index > numBins-1){
                cout<<"Not possible. g_index value = "<<g_index<<endl;
            }
            rgChromaticity[r_index][g_index] = rgChromaticity[r_index][g_index]+1;
        }
    }
    int rgCountsum=0;
    for(int i=0;i<numBins;i++){
        for(int j=0;j<numBins;j++){
            featureVector.push_back(rgChromaticity[i][j]);
            rgCountsum += rgChromaticity[i][j];
        }
    }
    if(rgCountsum != numRows*numCols){
        cout<<"Not Possible. rgCountsum = "<<rgCountsum<<" image single channel size = "<<numRows*numCols<<endl;
        exit(-1);
    }
    return 0;
}

/**
 * @brief calculates two rg chromaticity feature vectorfor given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the two rg chromaticityfeature results of the given imagee.
 * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int getTwoRGChromaticityFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){

    int numRows, numCols, numChannels;
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();
    Mat topHalf = Mat::zeros(numRows/2, numCols, src.type());
    Mat bottomHalf = Mat::zeros(numRows/2, numCols, src.type());
    Vec3b pixelVal;
    for(int i=0;i<numRows/2;i++){
        for(int j=0;j<numCols;j++){
            pixelVal = src.at<Vec3b>(i, j);
            for(int c=0; c< numChannels;c++){
                topHalf.at<Vec3b>(i, j)[c] = pixelVal[c];
                if(topHalf.at<Vec3b>(i, j)[c] != src.at<Vec3b>(i, j)[c] ){
                    cout<<"Not possible. Should eb same. topHalf.at<Vec3b>(i, j)[c] = "<<topHalf.at<Vec3b>(i, j)[c]<<"src.at<Vec3b>(i, j)[c] = "<<src.at<Vec3b>(i, j)[c]<<endl;
                }
            }
        }
    }
    
    int x = 0;
    for(int i=numRows/2;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            pixelVal = src.at<Vec3b>(i, j);
            for(int c=0; c< numChannels;c++){
                bottomHalf.at<Vec3b>(x, j)[c] = pixelVal[c];
                if(bottomHalf.at<Vec3b>(x, j)[c] != src.at<Vec3b>(i, j)[c] ){
                    cout<<"Not possible. Should eb same. bottomHalf.at<Vec3b>(i, j)[c] = "<<bottomHalf.at<Vec3b>(x, j)[c]<<"src.at<Vec3b>(i, j)[c] = "<<src.at<Vec3b>(i, j)[c]<<endl;
                }
            }
        }
        x++;
    }

    getRGChromaticityFeatureVector(topHalf,featureVector,numBins);
    getRGChromaticityFeatureVector(bottomHalf,featureVector,numBins);

    return 0;
}

/**
 * @brief calculates color and texture feature vector for given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the color and texture results of the given imagee.
 * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int getColorTextureFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){

    int numRows, numCols, numChannels;
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();
    getRGChromaticityFeatureVector(src,featureVector,numBins);
    Mat sobelImg;
    getMSobel(src, sobelImg);
    getRGChromaticityFeatureVector(sobelImg,featureVector,numBins);
    return 0;
}


/**
 * @brief calculates sunset feature vector scores for given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the sunset scores results of the given imagee.
 * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int getSunsetScoreFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){

    int min_hue = 0, max_hue = 30, min_saturation = 100, max_saturation = 255, min_value = 100, max_value = 255;
    int hueValue,saturationValue,colorValue;
    int numRows, numCols, numChannels;
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();
    int totalPixels = numRows*numCols;
    float sunsetScore = 0;
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            hueValue = src.at<Vec3b>(i, j)[0];
            saturationValue = src.at<Vec3b>(i, j)[1];
            colorValue = src.at<Vec3b>(i, j)[2];
            if(hueValue >= min_hue && hueValue <= max_hue && saturationValue >= min_saturation && saturationValue <= max_saturation && colorValue >= min_value && colorValue <= max_value){
                sunsetScore++;
            }
        }
    }
    sunsetScore = sunsetScore/totalPixels;
    featureVector.push_back(sunsetScore);
    return 0;
}

/**
 * @brief calculates sunset feature vector w.r.t intensity scores for given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the sunset scores w.r.t intensity results of the given imagee.
 * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int getSunsetIntensityScoreFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){

    int min_hue = 0, max_hue = 30, min_saturation = 100, max_saturation = 255, min_value = 100, max_value = 255;
    int hueValue,saturationValue,colorValue;
    int numRows, numCols, numChannels;
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();
    int totalPixels = numRows*numCols;
    float sunsetScore = 0;
    int intensity_index;
    int intensityHistogram[numBins];
    for(int i=0;i<numBins;i++){
            intensityHistogram[i] = 0;
    }
    for(int i=0;i<numRows;i++){
        for(int j=0;j<numCols;j++){
            hueValue = src.at<Vec3b>(i, j)[0];
            saturationValue = src.at<Vec3b>(i, j)[1];
            colorValue = src.at<Vec3b>(i, j)[2];
            if(hueValue >= min_hue && hueValue <= max_hue && saturationValue >= min_saturation && saturationValue <= max_saturation && colorValue >= min_value && colorValue <= max_value){
                sunsetScore++;
                double normalizeSaturationValue = (double)(saturationValue - min_saturation)/155.0;
                if(normalizeSaturationValue > 1){
                    cout<<"not possible. normalizeSaturationValue value = "<<normalizeSaturationValue<<endl;
                }
                intensity_index = (int)((normalizeSaturationValue )*(numBins-1)+0.5);
                if(intensity_index < 0 || intensity_index >= numBins){
                    cout<<"not possible. intensity index = "<<intensity_index<<endl;
                }
                intensityHistogram[intensity_index] = intensityHistogram[intensity_index] + 1; 
            }
        }
    }
    sunsetScore = sunsetScore/totalPixels;
    featureVector.push_back(sunsetScore);
    for(int i=0; i<numBins;i++){
        featureVector.push_back(intensityHistogram[i]);
    }
    return 0;
}

/**
 * @brief calculates sunset feature vector w.r.t split image row wise scores for given image. 
 * @param[in] src The input image.
 * @param[in] featureVector The vectore which will contain the sunset scores w.r.t intensity results of the given imagee.
 * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int getSunsetSplitScoreFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins){

    int min_hue = 0, max_hue = 30, min_saturation = 100, max_saturation = 255, min_value = 100, max_value = 255;
    int hueValue,saturationValue,colorValue;
    int numRows, numCols, numChannels;
    int numSplits = 10;

    // Mat src;
    // cv::cvtColor(inpImg, src, cv::COLOR_BGR2HSV);
    numRows = src.rows;
    numCols = src.cols;
    numChannels = src.channels();
    int totalPixels = numRows*numCols;
    float sunsetScore = 0;
    int totalIterations = numSplits;
    int iterationRows = numRows/10;
    float averageSaturationValue=0;
    std::vector<float> averageSaturationVector;
    int rowIterationCount=1;

    for(int i=0;i<numRows;i++){
        if(i%iterationRows == 0 && i != 0){
            if(sunsetScore > 0){
                averageSaturationValue = averageSaturationValue/sunsetScore;
            }
            sunsetScore = sunsetScore/(iterationRows*numCols);
            featureVector.push_back(sunsetScore);
            averageSaturationVector.push_back(averageSaturationValue);
            // cout<<rowIterationCount<<". i = "<<i<<" averageSaturationValue = "<<averageSaturationValue<<"sunsetScore  = "<<sunsetScore<<endl;
            sunsetScore = 0;
            averageSaturationValue = 0;
            rowIterationCount++;
            if(averageSaturationValue >255 || averageSaturationValue <0 || sunsetScore >1 || sunsetScore <0){
                cout<<"Not possible. averageSaturationValue = "<<averageSaturationValue<<" sunsetScore = "<<sunsetScore<<endl;
            }   
        }
        for(int j=0;j<numCols;j++){
            hueValue = src.at<Vec3b>(i, j)[0];
            saturationValue = src.at<Vec3b>(i, j)[1];
            colorValue = src.at<Vec3b>(i, j)[2];
            if(hueValue >= min_hue && hueValue <= max_hue && saturationValue >= min_saturation && saturationValue <= max_saturation && colorValue >= min_value && colorValue <= max_value){
                sunsetScore++;
                averageSaturationValue += saturationValue; 
            }
        }
    }

    for(int i=0;i<averageSaturationVector.size();i++){
        featureVector.push_back(averageSaturationVector[i]);
    }

    return 0;
}


/**
 * @brief calculates feature vector and stores all the feature values into a csv file. 
 * @param[in] image_paths The path where the images are present
 * @param[in] featuresFileName The path where feature vector csv file needs to be stored
 * @param[in] getFeatureVector function containing the logic of caluclating feature vector values.
 * * @param[in] numBins number of bins, the histogram is split into.
 * @return 0.
*/
int storeAllFeatureVectors(std::vector<std::string> &image_paths, String featuresFileName, FeatureVectorFunction getFeatureVector, int numBins){
    Mat src;
    string imagePath, imageName;
    int reset_file=1;
    int count=0;
    for(int i=0;i<image_paths.size();i++){
        
        imagePath = image_paths[i];
        src = imread(image_paths[i]); // read image fro the given filepath
        //checks if the input image contains data
        if(src.data == NULL) { 
            printf("Unable to read image ");
            exit(-1);
        }
        std::vector<float> featureVector;
        getFeatureVector(src, featureVector, numBins);
        imageName = imagePath.substr(imagePath.find_last_of("/")+1);
        append_image_data_csv(&featuresFileName[0], &imageName[0],  featureVector, reset_file);
        reset_file = 0;
        count++;
    }

    return 0;
}


/**
 * @brief given the feature vector and target image, finds the top images matching the target image
 * @param[in] targetImageName The target image name
 * @param[in] fileNames the image names
 * @param[in] data feature vectore of images data
 * @param[in] topN ftop n matches required
 * @param[in] getDistanceScore function to calculate the distance between the feature vectors
 * @param[in] isAscendingRank fis rank considerd in ascending order or descending order.
 * @param[in] bestMatchIndexes contains image names of the best matched
 * @return 0.
*/
int imageMatching(string targetImageName, std::vector<char*> fileNames, std::vector<std::vector<float>> data, int topN, DistanceScoreFunction getDistanceScore, bool isAscendingRank, std::vector<string> &bestMatchIndexes){

    int targetImageIndex;
    int totalFiles = fileNames.size();
    bool foundTargetFile = false;
    std::vector<float> targetImageVector;
    cout<<"Target Image name = "<<targetImageName<<endl;
    cout<<"totalFiles = "<<totalFiles<<endl;
    for(int i=0;i<totalFiles;i++){
        
        if(fileNames[i] == targetImageName){
            targetImageIndex = i;
            targetImageVector = data[i];
            foundTargetFile = true;
        }
    }
    if(!foundTargetFile){
        cout<<"Unable to fin the target Image"<<endl;
        exit(-1);
    }

    std::vector<float> imageFilterVector;

    //iterate over all images and get sum of squared difference
    double distanceVal;
    std::vector<double> distanceVector;
    for(int i=0;i<totalFiles;i++){
        imageFilterVector = data[i];
        distanceVal = getDistanceScore(targetImageVector, imageFilterVector);
        distanceVector.push_back(distanceVal);
    }
    vector<pair<double, int> > sortedDistances;
    sortDistances(distanceVector, sortedDistances);
    if(isAscendingRank){
        for (int i = 0; i < topN; i++) {
            cout<<i<<". "<<fileNames[sortedDistances[i].second]<<endl;
            bestMatchIndexes.push_back(fileNames[sortedDistances[i].second]);
        }
    }else{
        int n = sortedDistances.size();
        for (int i =n-1; i >= n - topN - 1; i--) {
            cout<<i<<". "<<fileNames[sortedDistances[i].second]<<endl;
            bestMatchIndexes.push_back(fileNames[sortedDistances[i].second]);
        }
    }
    return 0;
}


/**
 * @brief given the feature vector, finds the top ranked images w.r.t givenrequirement
 * @param[in] fileNames the image names
 * @param[in] data feature vectore of images data
 * @param[in] topN ftop n matches required
 * @param[in] getrankScore function to calculate the rank between the feature vectors
 * @param[in] isAscendingRank fis rank considerd in ascending order or descending order.
 * @param[in] bestMatchIndexes contains image names of the best matched
 * @return 0.
*/
int imageRanking(std::vector<char*> fileNames, std::vector<std::vector<float>> data, int topN, RankScoreFunction getrankScore, bool isAscendingRank, std::vector<string> &bestMatchIndexes){
    
    int totalFiles = fileNames.size();
    std::vector<float> targetImageVector;
    cout<<"totalFiles = "<<totalFiles<<endl;

    std::vector<float> imageFilterVector;

    //iterate over all images and get sum of squared difference
    double distanceVal;
    std::vector<double> distanceVector;
    for(int i=0;i<totalFiles;i++){
        imageFilterVector = data[i];
        distanceVal = getrankScore(imageFilterVector);
        distanceVector.push_back(distanceVal);
    }
    vector<pair<double, int> > sortedDistances;
    sortDistances(distanceVector, sortedDistances);
    if(isAscendingRank){
        for (int i = 0; i < topN; i++) {
            cout<<i<<". "<<fileNames[sortedDistances[i].second]<<endl;
            bestMatchIndexes.push_back(fileNames[sortedDistances[i].second]);
        }
    }else{
        int n = sortedDistances.size();
        for (int i =n-1; i >= n - topN - 1; i--) {
            cout<<i<<". "<<fileNames[sortedDistances[i].second]<<endl;
            bestMatchIndexes.push_back(fileNames[sortedDistances[i].second]);
        }
    }
    return 0;
}