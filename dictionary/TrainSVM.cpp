#include<iostream>
#include<fstream>
#include<string>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/opencv.hpp>

using namespace std;

int length;
int typeImage;
map<string,cv::Mat> createTraingImage(vector<string> filenames,vector<string> labels,cv::Mat vocabulary)
{
    cout<<"create training entered"<<endl;
    string filename;
    cv::Mat trainingImage;
    cv::Mat image;
    vector<cv::KeyPoint> keypoint;
    cv::Mat descriptor;
    string label;
    map<string,cv::Mat> mapedImage;
    cv::Ptr<cv::FeatureDetector> siftFeatureDetector(new cv::SiftFeatureDetector(300));
    cv::Ptr<cv::DescriptorExtractor> siftDescriptorExtractor(new cv::SiftDescriptorExtractor);
    cv::Ptr<cv::DescriptorMatcher> flannDescriptorMatcher(new cv::FlannBasedMatcher);
    cv::BOWImgDescriptorExtractor bowImgDescriptorExtractor(siftDescriptorExtractor,flannDescriptorMatcher);
    bowImgDescriptorExtractor.setVocabulary(vocabulary);
    cout<<"initialized fine"<<endl;
    cout<<filenames.size()<<endl;
    cout<<labels.size()<<endl;
    for(int i=0;i<filenames.size();i++)
    {
        filename=filenames[i];
        label=labels[i];
        image=cv::imread(filename);
        siftFeatureDetector->detect(image,keypoint);
        bowImgDescriptorExtractor.compute(image,keypoint,descriptor);
        mapedImage[label].create(0,descriptor.cols,descriptor.type());
        mapedImage[label].push_back(descriptor);

    }
    
    length=descriptor.cols;
    typeImage=descriptor.type();
    return mapedImage;

}
void trainSVMFinal(map<string,cv::Mat> mapedImage)
{
    cout<<"train SVM Final entered"<<endl;
    cv::Mat svmTrainImage;
    cv::SVMParams svmParam;
    cv::Mat labelsMat(0,1,CV_32FC1);
    cv::Mat samples(0,length,typeImage);
    string stringLabel;
    for(map<string,cv::Mat>::iterator it=mapedImage.begin(); it!=mapedImage.end(); it++)
    {
        stringLabel=it->first;
        if(it->first=="cat")
        {
            cv::Mat label=cv::Mat::ones(mapedImage[stringLabel].rows,1,CV_32FC1);
            labelsMat.push_back(label);
            samples.push_back(mapedImage[stringLabel]);

        }
        if(it->first=="dog")
        {
            cv::Mat label=cv::Mat::zeros(mapedImage[stringLabel].rows,1,CV_32FC1);
            labelsMat.push_back(label);
            samples.push_back(mapedImage[stringLabel]);
        }
    }
    cout<<"successfully labeled sampled the images"<<endl;
    svmParam.svm_type=cv::SVM::C_SVC;
    svmParam.kernel_type=cv::SVM::LINEAR;
    cv::TermCriteria term(cv::TermCriteria::COUNT,100,1e-6);
    svmParam.term_crit=term;
    cout<<"parameters created successfully"<<endl;
    cv::SVM svm;
    bool trainSvm=svm.train(samples,labelsMat,cv::Mat(),cv::Mat(),svmParam);
    cout<<"trained successfully "<<endl;
     char* fs="svmtrained.yml";
    
    svm.save(fs);

}


void trainSVM()
{
    cv::FileStorage fileStorage("dictionary.yml",cv::FileStorage::READ);
    cv::Mat vocabulary;
    fileStorage["vocabulary"]>>vocabulary;
    fileStorage.release();

    char* filename=new char[100];
    vector<string> filenames;
    vector<string> labels;

    for(int i=0; i<40; i++)
    {

        sprintf(filename,"%i.jpg",i+1);
        filenames.push_back(filename);
        if(i<16)
            labels.push_back("cat");
        if(i>=16&&i<=40)
            labels.push_back("dog");

    }
    cout<<"label added successfully"<<endl;
    map<string,cv::Mat> mapedImage=createTraingImage(filenames,labels,vocabulary);
    trainSVMFinal(mapedImage);
}
