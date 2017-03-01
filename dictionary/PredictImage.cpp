#include<iostream>
#include<string>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
cv::Mat loadVocabulary(cv::Mat predictImage)
{
    cout<<"load vocabulary"<<endl;
    vector<cv::KeyPoint> predictKeypoint;
    cv::Mat predictImageDescriptor;
    cv::FileStorage fileStorage("C:\\Users\\what\\Documents\\CodeBlock\\dictionary.yml",cv::FileStorage::READ);
    cv::Mat vocabulary;
    fileStorage["vocabulary"]>>vocabulary;
    fileStorage.release();
    cv::Ptr<cv::FeatureDetector> siftFeatureDetector(new cv::SiftFeatureDetector(300));
    cv::Ptr<cv::DescriptorExtractor> siftDescriptorExtractor(new cv::SiftDescriptorExtractor);
    cv::Ptr<cv::DescriptorMatcher> flannBasedMatcher(new cv::FlannBasedMatcher);
    cv::BOWImgDescriptorExtractor bowImgDescriptorExtractor(siftDescriptorExtractor,flannBasedMatcher);
    bowImgDescriptorExtractor.setVocabulary(vocabulary);
    siftFeatureDetector->detect(predictImage,predictKeypoint);
    if(predictKeypoint.empty())
        cout<<"no keypoint "<<endl;
    bowImgDescriptorExtractor.compute(predictImage,predictKeypoint,predictImageDescriptor);
    cout<<"load vocabulary return"<<endl;
    if(predictImageDescriptor.empty())
        cout<<"the no descriptor extracted"<<endl;
    return predictImageDescriptor;

}

void predict(string imageName)
{
    cout<<"predict"<<endl;
    cv::Mat predictMat=cv::imread(imageName);
    if(predictMat.empty())
        cout<<"image is not read"<<endl;
    cv::SVM svm;
    svm.load("C:\\Users\\what\\Documents\\CodeBlock\\svmtrained.yml");
    cout<<"svm loaded successfully"<<endl;
    cv::Mat predictImageDescriptor=loadVocabulary(predictMat);
    int result=(int)svm.predict(predictImageDescriptor);
    cout<<"predicted successfully"<<endl;
    if(result==1)
        cout<<"inputed Image is cat"<<endl;
    if(result==0)
        cout<<"inputed Image is dog"<<endl;
}
