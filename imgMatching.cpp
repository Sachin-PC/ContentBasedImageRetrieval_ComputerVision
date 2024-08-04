/*
    Sachin Palahalli Chandrakumar
    Spring 2024
    Content Based Image Detection using different feature vectors
*/

#include <cstdio>
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Defining the typedef of methods used as function parameters in this file
typedef int (*FeatureVectorFunction)(const cv::Mat&, std::vector<float>&, int);
typedef double (*DistanceScoreFunction)(std::vector<float>, std::vector<float>);
typedef double (*RankScoreFunction)(std::vector<float>);

// Defining the prototypes of methods used in this file
int get_image_paths( std::vector<string> &image_paths);
int read_image_data_csv( char *filename, std::vector<char *> &filenames, std::vector<std::vector<float>> &data, int echo_file );

int storeAllFeatureVectors(std::vector<std::string> &image_paths, String featuresFileName, FeatureVectorFunction getFeatureVector, int numBins);
int getBaselineMatchinFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins);
int getRGChromaticityFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins);
int getTwoRGChromaticityFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins);
int getColorTextureFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins);
int getSunsetScoreFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins);
int getSunsetSplitScoreFeatureVector(const cv::Mat& inpImg, std::vector<float> &featureVector, int numBins);
int getSunsetIntensityScoreFeatureVector(const cv::Mat& src, std::vector<float> &featureVector, int numBins);
int imageMatching(string targetImageName, std::vector<char*> fileNames, std::vector<std::vector<float>> data, int topN, DistanceScoreFunction getDistanceScore, bool isAscendingRank, std::vector<string> &bestMatchIndexes);
double getSumSquaredDifference(vector<float>targetImageVector, std::vector<float> imageFilterVector);
double gethistogramIntersection(vector<float>targetImageVector, std::vector<float> imageFilterVector);
double getSunsetScores(std::vector<float> imageFilterVector);
double getSunsetIntensityScore(std::vector<float> imageFilterVector);
double getSunsetIntensityHistogramScore(std::vector<float> imageFilterVector);
int imageRanking(std::vector<char*> fileNames, std::vector<std::vector<float>> data, int topN, RankScoreFunction getrankScore, bool isAscendingRank, std::vector<string> &bestMatchIndexes);



/**
 * @brief reads all images csv files and gets the feature vector data.
 * This function take the path where the image feature data is present and then 
 * loads all the data present in the file to memory
 * @param[in] featuresFileName The csv file name.
 * @param[in] fileNames The list of all image names.
 * @param[in] data The 2D list of all images feature vector data.
 * @return 0.
*/
int getImagesFeaturesData(string featuresFileName, std::vector<char*> &fileNames, std::vector<std::vector<float>> &data){
    
    int echo_file = 0;
    read_image_data_csv( &featuresFileName[0], fileNames, data, echo_file );
    int totalFiles = fileNames.size();
    int totalData = data.size();
    if(totalFiles != totalData){
        cout<<" Number of Image names and the filter vectores present sould be same!";
        exit(-1);
    }
    return 0;
}

/**
 * @brief General method to calculate feature vector and store it.
 * In this method, all images feature vectors are calculated and then 
 * stored in a specific file location as csv file
 * @param[in] featuresFileName The csv file name.
 * @param[in] featureVectorFunction function which contains the logic of feature calculation.
 * @param[in] numBins Histogram Bins considered.
 * @return 0.
*/
int storeFeatureVectors(string featuresFileName, FeatureVectorFunction featureVectorFunction, int numBins){
    std::vector<string> image_paths;
    get_image_paths(image_paths);
    storeAllFeatureVectors(image_paths, featuresFileName, featureVectorFunction, numBins);
    return 0;
}

/**
 * @brief Given a target images, gets the top images matched
 * @param[in] featuresFileName The csv file name.
 * @param[in] targetImageName The target image considered.
 * @param[in] topN The top results required.
 * @param[in] bestMatchIndexes The vector containing the image names of topN images.
 * @param[in] distanceScoreFunction function which contains the logic of distance calculation.
 * @param[in] isAscendingRank is results ranked in ascending order or descending order.
 * @return 0.
*/
int getTopMatchedImages(string featuresFileName, string targetImageName, int topN, std::vector<string> &bestMatchIndexes, DistanceScoreFunction distanceScoreFunction, bool isAscendingRank){
    std::vector<char*> fileNames;
    std::vector<std::vector<float>> data;
    getImagesFeaturesData(featuresFileName, fileNames, data);
    imageMatching(targetImageName,fileNames, data, topN, distanceScoreFunction, isAscendingRank, bestMatchIndexes);
    return 0;
}

/**
 * @brief gets the top ranked images based on requirement.
 * @param[in] featuresFileName The csv file name.
 * @param[in] topN The top results required.
 * @param[in] bestMatchIndexes The vector containing the image names of topN images.
 * @param[in] getrankScore function which contains the logic of rank calculation.
 * @param[in] isAscendingRank is results ranked in ascending order or descending order.
 * @return 0.
*/
int getRankedImages(string featuresFileName, int topN, std::vector<string> &bestMatchIndexes, RankScoreFunction getrankScore, bool isAscendingRank){
    std::vector<char*> fileNames;
    std::vector<std::vector<float>> data;
    getImagesFeaturesData(featuresFileName, fileNames, data);
    // imageMatching(targetImageName,fileNames, data, topN, distanceScoreFunction, isAscendingRank, bestMatchIndexes);
    imageRanking(fileNames, data, topN, getrankScore, isAscendingRank, bestMatchIndexes);
    return 0;
}


int main(int argc, char *argv[]){

    char targetImageName[256];

    if(argc != 4){
        printf("Incorrect Command Line input. Usage: ");
        exit(-1);
    }
    strcpy(targetImageName, argv[1]);

    // command line comments map
    std::unordered_map<string,int> command_map;
    command_map["store_baseline_m"] = 1;
    command_map["rank_baseline_m"] = 2;
    command_map["store_rghistogram_m"] = 3;
    command_map["rank_rghistogram_m"] = 4;
    command_map["store_two_rghistogram_m"] = 5;
    command_map["rank_two_rghistogram_m"] = 6;
    command_map["store_color_texture_m"] = 7;
    command_map["rank_color_texture_m"] = 8;
    command_map["dnn_embeddings_m"] = 9;
    command_map["store_sunset_m"] = 10;
    command_map["rank_sunset_m"] = 11;
    command_map["store_sunsetsplit_m"] = 12;
    command_map["rank_sunsetsplit_m"] = 13;
    command_map["store_sunsetintensity_m"] = 14;
    command_map["rank_sunsetintensity_m"] = 15;
    command_map["rank_sunsetintensitytarget_m"] = 16;
    command_map["rank_sunset_m_least"] = 17;
    

    int command = command_map[argv[2]];
    int topN = std::stoi(argv[3]);
    bool displayResults = false;
    string featuresFileName, colorFeaturesFileName, textureFeaturesFileName;
    std::vector<string> bestMatchIndexes;
    char matchingType[256];
    int numBins=0;
    bool isAscendingRank = true;
    //switch case to consider all command line inputs and redirecting to perform required operations.
    switch(command){
        case 1:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/baselineFeaturesFile.csv";
            storeFeatureVectors(featuresFileName, &getBaselineMatchinFeatureVector, 0);
            break;
        case 2:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/baselineFeaturesFile.csv";
            isAscendingRank = true;
            getTopMatchedImages(featuresFileName, targetImageName, topN, bestMatchIndexes, &getSumSquaredDifference, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Baseline Matching");
            break;
        case 3:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/rgChromaticityFeaturesFile.csv";
            numBins = 16;
            storeFeatureVectors(featuresFileName, &getRGChromaticityFeatureVector, numBins);
            break;
        case 4:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/rgChromaticityFeaturesFile.csv";
            isAscendingRank = false;
            getTopMatchedImages(featuresFileName, targetImageName, topN, bestMatchIndexes, &gethistogramIntersection, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "RG Chromaticity Histogram Matching");
            break;
        case 5:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/TworgChromaticityFeaturesFile.csv";
            numBins = 16;
            storeFeatureVectors(featuresFileName, &getTwoRGChromaticityFeatureVector, numBins);
            break;
        case 6:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/TworgChromaticityFeaturesFile.csv";
            isAscendingRank = false;
            getTopMatchedImages(featuresFileName, targetImageName, topN, bestMatchIndexes, &gethistogramIntersection, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;
        case 7:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/ColorTextureFeaturesFile.csv";
            numBins = 16;
            storeFeatureVectors(featuresFileName, &getColorTextureFeatureVector, numBins);
            break;
        case 8:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/ColorTextureFeaturesFile.csv";
            isAscendingRank = false;
            getTopMatchedImages(featuresFileName, targetImageName, topN, bestMatchIndexes, &gethistogramIntersection, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;
        case 9:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/ResNet18_olym.csv";
            isAscendingRank = true;
            getTopMatchedImages(featuresFileName, targetImageName, topN, bestMatchIndexes, &getSumSquaredDifference, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Baseline Matching");
            break;
        case 10:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetScoresFile.csv";
            numBins = 16;
            storeFeatureVectors(featuresFileName, &getSunsetScoreFeatureVector, numBins);
            break;
        case 11:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetScoresFile.csv";
            isAscendingRank = false;
            getRankedImages(featuresFileName, topN, bestMatchIndexes, &getSunsetScores, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;
        case 12:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetSplitScoresVectorFile.csv";
            numBins = 16;
            storeFeatureVectors(featuresFileName, &getSunsetSplitScoreFeatureVector, numBins);
            break;
        case 13:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetSplitScoresVectorFile.csv";
            isAscendingRank = false;
            getRankedImages(featuresFileName, topN, bestMatchIndexes, &getSunsetIntensityScore, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;
        case 14:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetIntensityScoresVectorFile.csv";
            numBins = 16;
            storeFeatureVectors(featuresFileName, &getSunsetIntensityScoreFeatureVector, numBins);
            break;
        case 15:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetIntensityScoresVectorFile.csv";
            isAscendingRank = false;
            getRankedImages(featuresFileName, topN, bestMatchIndexes, &getSunsetIntensityHistogramScore, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;
        case 16:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetIntensityScoresVectorFile.csv";
            isAscendingRank = false;
            getTopMatchedImages(featuresFileName, targetImageName, topN, bestMatchIndexes, &gethistogramIntersection, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;
        case 17:
            featuresFileName = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/featureFiles/sunsetScoresFile.csv";
            isAscendingRank = true;
            getRankedImages(featuresFileName, topN, bestMatchIndexes, &getSunsetScores, isAscendingRank);
            displayResults = true;
            strcpy(matchingType, "Two RG Chromaticity Histogram Matching");
            break;

    }


    //display the top result images
    if(displayResults){
        std::vector<string>  outputImageName;
        string outputImageDir;
        namedWindow(matchingType, 1);
        string dirname = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/olympus";
        strcat(matchingType, targetImageName);
        Mat srcImage;
        std::vector<cv::Mat> resultantImages;
        for(int i=0;i<topN;i++){
            string imgFileName = dirname;
            imgFileName.append("/");
            imgFileName.append(bestMatchIndexes[i]);
            resultantImages.push_back(imread(imgFileName));
            outputImageName.push_back(bestMatchIndexes[i]);
        }

        for(int i=0;i<topN;i++){
            outputImageDir =  "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth Semester/PRCV/Projects/2/utilities/saved_photos/";
            imwrite(outputImageDir+outputImageName[i], resultantImages[i]);
            cv::imshow(matchingType, resultantImages[i]);
            waitKey(0);
        }
        destroyWindow(matchingType);   
    }
    printf("Terminating\n");
    return(0);
}



