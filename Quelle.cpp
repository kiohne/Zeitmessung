#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <ppl.h>
#include <windows.h>
#include <stdio.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\core\cvstd.hpp>

#include <opencv2\core\cuda.hpp>
#include <opencv2\cudawarping.hpp>
#include <opencv2\cudaarithm.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <opencv2\cudafilters.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

namespace
{

    void writeTextFile(string fileName, string text)
    {
        std::string str;
        char c;
        std::ifstream file(fileName, std::ios::in);
        while (file.get(c)) {
            str.push_back(c);
        }
        std::ofstream out(fileName);
        out << str << endl << text;
        return;
    }
    std::vector<double> readFileDataToVector(string filename, int numLines)
    {
        //Outputvector
        std::vector<double> fileData(numLines);

        // check if file exist

        std::fstream inFile;
        inFile.open(filename, std::ios::in);

        if (inFile.fail()) {
            fileData.resize(0);
            return fileData;
        }

        for (int i = 0; i < numLines; i++)
        {
            double value;
            inFile >> value;
            fileData[i] = value;
        }

        return fileData;
    }
    cv::cuda::GpuMat scaleImageForCentralProjectioncAndShadingCorrection(cv::cuda::GpuMat& d_img, cv::Point2f princPoint, double beta, double sc)
    {

        ////1. central projection
        cv::Mat transformMatrix = cv::getRotationMatrix2D(princPoint, 0.0, beta);
        cv::cuda::GpuMat d_transformedImg(d_img.size(), d_img.type());
        cv::cuda::GpuMat d_dst;

        cv::cuda::warpAffine(d_img, d_transformedImg, transformMatrix, d_img.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        //2. Shading correction
        cv::cuda::multiply(d_transformedImg, cv::Scalar(sc, sc, sc), d_dst);

        return d_dst;

    }
    cv::cuda::GpuMat scaleImageForCentralProjectioncAndShadingCorrectionSmallScale(cv::cuda::GpuMat& d_img, cv::Point2f princPoint, double beta, double sc)
    {

        ////1. central projection
        auto begin = std::chrono::high_resolution_clock::now();
        cv::Mat transformMatrix = cv::getRotationMatrix2D(princPoint, 0.0, beta);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 10WallTimeMat transformMatrix = getRotationMatrix2D(princPoint, 0.0, beta)";
        writeTextFile("10GPUscaleImageForCentralProjectioncAndShadingCorrectionSmallScale.txt", text);


        cv::cuda::GpuMat d_transformedImg(d_img.size(), d_img.type());
        cv::cuda::GpuMat d_dst;

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::warpAffine(d_img, d_transformedImg, transformMatrix, d_img.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 11WallTimeWarpAffine(d_img, d_transformedImg, transformMatrix, d_img.size(), INTER_CUBIC, BORDER_REPLICATE)";
        writeTextFile("10GPUscaleImageForCentralProjectioncAndShadingCorrectionSmallScale.txt", text);


        //2. Shading correction
        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::multiply(d_transformedImg, cv::Scalar(sc, sc, sc), d_dst);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 12WallTimeMultiply(d_transformedImg, Scalar(sc, sc, sc), d_dst)";
        writeTextFile("10GPUscaleImageForCentralProjectioncAndShadingCorrectionSmallScale.txt", text);

        return d_dst;

    }
    cv::cuda::GpuMat calculateSharpness(cv::cuda::GpuMat& d_img, cv::Mat& gauss_kernelSigma4, cv::Mat& gauss_kernelSigma8)
    {
        int filtCoreSz = 33;
        cv::cuda::GpuMat d_grayImg, d_grayImgFloat;


        cv::cuda::cvtColor(d_img, d_grayImg, cv::COLOR_BGR2GRAY);

        d_grayImg.convertTo(d_grayImgFloat, CV_32F);

        cv::cuda::GpuMat d_lf(d_grayImgFloat.size(), d_grayImgFloat.type());
        cv::cuda::GpuMat d_hf(d_lf.size(), d_lf.type());
        cv::cuda::GpuMat d_sh(d_hf.size(), d_hf.type());

        cv::Ptr<cuda::Convolution> conv = cv::cuda::createConvolution();          //cv::Ptr< T > //Template class for smart pointers with shared ownership
        cv::mulTransposed(cv::getGaussianKernel(filtCoreSz, 4.0, CV_32F), gauss_kernelSigma4, false);

        conv->convolve(d_grayImgFloat, gauss_kernelSigma4, d_lf, true);

        cv::cuda::copyMakeBorder(d_lf, d_lf, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, BORDER_REPLICATE);

        cv::cuda::absdiff(d_grayImgFloat, d_lf, d_hf);

        filtCoreSz = 65;

        cv::mulTransposed(cv::getGaussianKernel(filtCoreSz, 8.0, CV_32F), gauss_kernelSigma8, false);

        conv->convolve(d_hf, gauss_kernelSigma8, d_sh, true);

        cv::cuda::copyMakeBorder(d_sh, d_sh, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, BORDER_REPLICATE);

        cv::cuda::divide(d_sh, d_lf, d_sh);

        return d_sh;
    }
    cv::cuda::GpuMat calculateSharpnessSmallScale(cv::cuda::GpuMat& d_img, cv::Mat& gauss_kernelSigma4, cv::Mat& gauss_kernelSigma8)
    {
        int filtCoreSz = 33;
        cv::cuda::GpuMat d_grayImg, d_grayImgFloat;

        auto begin = std::chrono::high_resolution_clock::now();
        cv::cuda::cvtColor(d_img, d_grayImg, cv::COLOR_BGR2GRAY);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 24WallTimeCvtColor(d_img, d_grayImg, COLOR_BGR2GRAY)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        d_grayImg.convertTo(d_grayImgFloat, CV_32F);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 25WallTimeD_grayImg.convertTo(d_grayImgFloat, CV_32F)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        cv::cuda::GpuMat d_lf(d_grayImgFloat.size(), d_grayImgFloat.type());
        cv::cuda::GpuMat d_hf(d_lf.size(), d_lf.type());
        cv::cuda::GpuMat d_sh(d_hf.size(), d_hf.type());

        begin = std::chrono::high_resolution_clock::now();
        cv::Ptr<cuda::Convolution> conv = cv::cuda::createConvolution();          //cv::Ptr< T > //Template class for smart pointers with shared ownership
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 26WallTimePtr<Convolution> conv = createConvolution()";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::mulTransposed(cv::getGaussianKernel(filtCoreSz, 4.0, CV_32F), gauss_kernelSigma4, false);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 27WallTimeMulTransposed(getGaussianKernel(filtCoreSz, 4.0, CV_32F)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        conv->convolve(d_grayImgFloat, gauss_kernelSigma4, d_lf, true);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 28WallTimeConv-convolve(d_grayImgFloat, gauss_kernelSigma4, d_lf, true)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::copyMakeBorder(d_lf, d_lf, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, BORDER_REPLICATE);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 29WallTimeCopyMakeBorder(d_lf, d_lf,)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::absdiff(d_grayImgFloat, d_lf, d_hf);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 30WallTimeAbsdiff(d_grayImgFloat, d_lf, d_hf)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        filtCoreSz = 65;

        begin = std::chrono::high_resolution_clock::now();
        cv::mulTransposed(cv::getGaussianKernel(filtCoreSz, 8.0, CV_32F), gauss_kernelSigma8, false);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 31WallTimeMulTransposed(d_grayImgFloat, d_lf, d_hf)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        conv->convolve(d_hf, gauss_kernelSigma8, d_sh, true);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 32WallTimeConvolve(d_hf, gauss_kernelSigma8, d_sh, true)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::copyMakeBorder(d_sh, d_sh, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, 0.5 * filtCoreSz, BORDER_REPLICATE);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 33WallTimeCopyMakeBorder(d_sh, d_sh,)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::divide(d_sh, d_lf, d_sh);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 34WallTimeDivide(d_sh, d_lf, d_sh)";
        writeTextFile("13GPUcalculateSharpnessSmallScale.txt", text);

        return d_sh;
    }
    cv::cuda::GpuMat calcSharpWithSobelDeviation(cv::cuda::GpuMat& d_img, int kernel)
    {

        cv::cuda::GpuMat d_grayImg, d_grayImgFloat, d_grad_x, d_grad_y, d_grad;

        cv::cuda::cvtColor(d_img, d_grayImg, cv::COLOR_BGR2GRAY);

        d_grayImg.convertTo(d_grayImgFloat, CV_32F);

        cv::Ptr<cv::cuda::Filter> Sobel_x = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, kernel);
        cv::Ptr<cv::cuda::Filter> Sobel_y = cv::cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, kernel);

        Sobel_x->apply(d_grayImgFloat, d_grad_x);
        Sobel_y->apply(d_grayImgFloat, d_grad_y);

        cv::cuda::abs(d_grad_x, d_grad_x);
        cv::cuda::abs(d_grad_y, d_grad_y);

        cv::cuda::addWeighted(d_grad_x, 0.5, d_grad_y, 0.5, 0, d_grad);


        return d_grad;
    }
    cv::cuda::GpuMat calcSharpWithSobelDeviationSmallScale(cv::cuda::GpuMat& d_img, int kernel)
    {

        cv::cuda::GpuMat d_grayImg, d_grayImgFloat, d_grad_x, d_grad_y, d_grad;

        auto begin = std::chrono::high_resolution_clock::now();
        cv::cuda::cvtColor(d_img, d_grayImg, cv::COLOR_BGR2GRAY);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 13WallTimeCvtColor(d_img, d_grayImg, COLOR_BGR2GRAY)";
        writeTextFile("11GPUalcSharpWithSobelDeviationSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        d_grayImg.convertTo(d_grayImgFloat, CV_32F);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 14WallTimeD_grayImg.convertTo(d_grayImgFloat, CV_32F)";
        writeTextFile("11GPUalcSharpWithSobelDeviationSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::Ptr<cv::cuda::Filter> Sobel_x = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, kernel);
        cv::Ptr<cv::cuda::Filter> Sobel_y = cv::cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, kernel);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 15WallTimePtr<Filter> Sobel = createSobelFilter(CV_32F, CV_32F, 1, 0, kernel)";
        writeTextFile("11GPUalcSharpWithSobelDeviationSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        Sobel_x->apply(d_grayImgFloat, d_grad_x);
        Sobel_y->apply(d_grayImgFloat, d_grad_y);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 16WallTimeSobel_x-apply(d_grayImgFloat, d_grad_x)";
        writeTextFile("11GPUalcSharpWithSobelDeviationSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::abs(d_grad_x, d_grad_x);
        cv::cuda::abs(d_grad_y, d_grad_y);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 17WallTimeAbs(d_grad_y, d_grad_y)";
        writeTextFile("11GPUalcSharpWithSobelDeviationSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::addWeighted(d_grad_x, 0.5, d_grad_y, 0.5, 0, d_grad);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 18WallTimeAddWeighted(d_grad_x, 0.5, d_grad_y, 0.5, 0, d_grad)";
        writeTextFile("11GPUalcSharpWithSobelDeviationSmallScale.txt", text);


        return d_grad;
    }
    cv::cuda::GpuMat calcSharpWithLaplace(cv::cuda::GpuMat& d_img)
    {
        cv::cuda::GpuMat d_grayImg, d_dst, d_abs_dst;
        cv::cuda::cvtColor(d_img, d_grayImg, cv::COLOR_BGR2GRAY);             // Convert the image to grayscale
        cv::cuda::GpuMat d_grayImgFloat;
        d_grayImg.convertTo(d_grayImgFloat, CV_32F);

        cv::Ptr<cv::cuda::Filter> Laplace = cv::cuda::createLaplacianFilter(CV_32F, CV_32F, 1); //ksize == 1 -> [0, 1, 0; 1, -4, 1; 0, 1, 1]

        Laplace->apply(d_grayImgFloat, d_dst);
        cv::cuda::abs(d_dst, d_abs_dst);
        return d_abs_dst;
    }
    cv::cuda::GpuMat calcSharpWithLaplaceSmallScale(cv::cuda::GpuMat& d_img)
    {
        cv::cuda::GpuMat d_grayImg, d_dst, d_abs_dst;

        auto begin = std::chrono::high_resolution_clock::now();
        cv::cuda::cvtColor(d_img, d_grayImg, cv::COLOR_BGR2GRAY);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 19WallTimeCvtColor(d_img, d_grayImg, COLOR_BGR2GRAY)";
        writeTextFile("12GPUcalcSharpWithLaplaceSmallScale.txt", text);
        // Convert the image to grayscale
        cv::cuda::GpuMat d_grayImgFloat;

        begin = std::chrono::high_resolution_clock::now();
        d_grayImg.convertTo(d_grayImgFloat, CV_32F);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 20WallTimeD_grayImg.convertTo(d_grayImgFloat, CV_32F)";
        writeTextFile("12GPUcalcSharpWithLaplaceSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::Ptr<cv::cuda::Filter> Laplace = cv::cuda::createLaplacianFilter(CV_32F, CV_32F, 1); //ksize == 1 -> [0, 1, 0; 1, -4, 1; 0, 1, 1]
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 21WallTimePtrFilter Laplace = createLaplacianFilter(CV_32F, CV_32F, 1)";
        writeTextFile("12GPUcalcSharpWithLaplaceSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        Laplace->apply(d_grayImgFloat, d_dst);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 22WallTimeLaplace-apply(d_grayImgFloat, d_dst)";
        writeTextFile("12GPUcalcSharpWithLaplaceSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::abs(d_dst, d_abs_dst);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 23WallTimeAbs(d_dst, d_abs_dst)";
        writeTextFile("12GPUcalcSharpWithLaplaceSmallScale.txt", text);

        return d_abs_dst;
    }
    cv::Mat scaleImageForCentralProjectionc(cv::Mat& img, cv::Point2f princPoint, double beta)
    {
        cv::Mat transformMatrix = cv::getRotationMatrix2D(princPoint, 0.0, beta);
        cv::Mat transformedImg(img.size(), img.type());
        cv::warpAffine(img, transformedImg, transformMatrix, img.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        return transformedImg;
    }
    cv::Mat shadingCorrection(cv::Mat& img, double sc)
    {
        return img.mul(cv::Scalar(sc, sc, sc));

    }
    cv::Mat calculateSharpness(cv::Mat& img)
    {
        //auto begin_calculateSharpness = std::chrono::high_resolution_clock::now();
        cv::Mat grayImg;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

        cv::Mat grayImgFloat;
        grayImg.convertTo(grayImgFloat, CV_32F);

        cv::Mat lf(grayImgFloat.size(), grayImgFloat.type());
        cv::GaussianBlur(grayImgFloat, lf, cv::Size(0, 0), 4.0);

        cv::Mat hf;
        cv::absdiff(grayImgFloat, lf, hf);

        cv::Mat sh(grayImgFloat.size(), grayImgFloat.type());
        cv::GaussianBlur(hf, sh, cv::Size(0, 0), 8.0);

        cv::divide(sh, lf, sh);


        return sh;
    }
    cv::Mat calcSharpWithSobelDeviation(cv::Mat& img)
    {

        cv::Mat grayImg;
        cv::Mat grad;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

        cv::Mat grayImgFloat;
        grayImg.convertTo(grayImgFloat, CV_32F);

        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        Sobel(grayImgFloat, grad_x, -1, 1, 0, -1);
        Sobel(grayImgFloat, grad_y, -1, 0, 1, -1);

        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        return grad;
    }
    cv::Mat calcSharpWithLaplace(cv::Mat& img)
    {
        Mat grayImg, dst;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        cv::Mat grayImgFloat;
        grayImg.convertTo(grayImgFloat, CV_32F);
        cv::Mat lf(grayImgFloat.size(), grayImgFloat.type());

        lf = grayImgFloat;

        Mat abs_dst;
        Laplacian(lf, dst, -1, 3);
        convertScaleAbs(dst, abs_dst);
        return abs_dst;
    }
    cv::Mat scaleImageForCentralProjectioncSmallScale(cv::Mat& img, cv::Point2f princPoint, double beta)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        cv::Mat transformMatrix = cv::getRotationMatrix2D(princPoint, 0.0, beta);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 66CPUGetRotationMatrix2D(princPoint, 0.0, beta)";
        writeTextFile("45CPUscaleImageForCentralProjectioncSmallScale.txt", text);
        cv::Mat transformedImg(img.size(), img.type());
        begin = std::chrono::high_resolution_clock::now();
        cv::warpAffine(img, transformedImg, transformMatrix, img.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 67CPUwarpAffine(img, transformedImg, transformMatrix)";
        writeTextFile("45CPUscaleImageForCentralProjectioncSmallScale.txt", text);
        return transformedImg;
    }
    cv::Mat shadingCorrectionSmallScale(cv::Mat& img, double sc)
    {
        auto begin = std::chrono::high_resolution_clock::now();
        return img.mul(cv::Scalar(sc, sc, sc));
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 68CPUreturn img.mul";
        writeTextFile("46CPUshadingCorrectionSmallScale.txt", text);

    }
    cv::Mat calculateSharpnessSmallScale(cv::Mat& img)
    {
        //auto begin_calculateSharpness = std::chrono::high_resolution_clock::now();
        cv::Mat grayImg;
        auto begin = std::chrono::high_resolution_clock::now();
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 78CPUcvtColor(img, grayImg, cv::COLOR_BGR2GRAY)";
        writeTextFile("49CPUcalculateSharpnessSmallScale.txt", text);
        cv::Mat grayImgFloat;
        begin = std::chrono::high_resolution_clock::now();
        grayImg.convertTo(grayImgFloat, CV_32F);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 79CPU grayImg.convertTo(grayImgFloat, CV_32F)";
        writeTextFile("49CPUcalculateSharpnessSmallScale.txt", text);
        cv::Mat lf(grayImgFloat.size(), grayImgFloat.type());
        begin = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(grayImgFloat, lf, cv::Size(0, 0), 4.0);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 80CPUcv::GaussianBlur(grayImgFloat, lf, cv::Size(0, 0), 4.0)";
        writeTextFile("49CPUcalculateSharpnessSmallScale.txt", text);

        cv::Mat hf;
        begin = std::chrono::high_resolution_clock::now();
        cv::absdiff(grayImgFloat, lf, hf);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 81CPUcv::absdiff(grayImgFloat, lf, hf)";
        writeTextFile("49CPUcalculateSharpnessSmallScale.txt", text);
        cv::Mat sh(grayImgFloat.size(), grayImgFloat.type());
        begin = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(hf, sh, cv::Size(0, 0), 8.0);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 82CPUcv::GaussianBlur(hf, sh, cv::Size(0, 0), 8.0)";
        writeTextFile("49CPUcalculateSharpnessSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        cv::divide(sh, lf, sh);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 83CPUcv::divide(sh, lf, sh)";
        writeTextFile("49CPUcalculateSharpnessSmallScale.txt", text);

        return sh;
    }
    cv::Mat calcSharpWithSobelDeviationSmallScale(cv::Mat& img)
    {

        cv::Mat grayImg;
        cv::Mat grad;
        auto begin = std::chrono::high_resolution_clock::now();
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 69CPUcvtColor(img, grayImg)";
        writeTextFile("47CPUcalcSharpWithSobelDeviationSmallScale.txt", text);
        cv::Mat grayImgFloat;
        begin = std::chrono::high_resolution_clock::now();
        grayImg.convertTo(grayImgFloat, CV_32F);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 70CPUgrayImg.convertTo(grayImgFloat, CV_32F)";
        writeTextFile("47CPUcalcSharpWithSobelDeviationSmallScale.txt", text);
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        begin = std::chrono::high_resolution_clock::now();
        Sobel(grayImgFloat, grad_x, -1, 1, 0, -1);
        Sobel(grayImgFloat, grad_y, -1, 0, 1, -1);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 71CPUSobel(grayImgFloat, grad_x, -1, 1, 0, -1)";
        writeTextFile("47CPUcalcSharpWithSobelDeviationSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 72CPUconvertScaleAbs(grad_y, abs_grad_y)";
        writeTextFile("47CPUcalcSharpWithSobelDeviationSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 73CPUaddWeighted(abs_grad_x)";
        writeTextFile("47CPUcalcSharpWithSobelDeviationSmallScale.txt", text);
        return grad;
    }
    cv::Mat calcSharpWithLaplaceSmallScale(cv::Mat& img)
    {
        Mat grayImg, dst;
        auto begin = std::chrono::high_resolution_clock::now();
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 74CPUcvtColor(img, grayImg, )";
        writeTextFile("48CPUcalcSharpWithLaplaceSmallScale.txt", text);
        cv::Mat grayImgFloat;
        begin = std::chrono::high_resolution_clock::now();
        grayImg.convertTo(grayImgFloat, CV_32F);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 75CPUgrayImg.convertTo(grayImgFloat, CV_32F)";
        writeTextFile("48CPUcalcSharpWithLaplaceSmallScale.txt", text);
        cv::Mat lf(grayImgFloat.size(), grayImgFloat.type());

        lf = grayImgFloat;

        Mat abs_dst;
        begin = std::chrono::high_resolution_clock::now();
        Laplacian(lf, dst, -1, 3);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 76CPULaplacian(lf, dst, -1, 3)";
        writeTextFile("48CPUcalcSharpWithLaplaceSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        convertScaleAbs(dst, abs_dst);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 77CPUconvertScaleAbs";
        writeTextFile("48CPUcalcSharpWithLaplaceSmallScale.txt", text);
        return abs_dst;
    }
    cv::Mat getSharpestPixel(cv::Mat& maxImg, cv::Mat& sh, cv::Mat& img, cv::Mat& edof)
    {
        cv::Mat maskImage;
        cv::compare(maxImg, sh, maskImage, cv::CMP_EQ);

        if (maskImage.type() != CV_8UC1)
        {
            maskImage.convertTo(maskImage, CV_8UC1);
        }
        return maskImage;
    }
}

int main()
{
    //Input range, numImages and resolution, idx of Images at F= 0 dpt
    double minF = -2.0;
    double resolution = 0.1;
    int numImages = 51;
    int idxZero = 20;

    stringstream bufferDeviceInfo;
    //Setup  GPU
    cv::cuda::setDevice(0);

    //add information about the machine
    int CPUInfo[4] = { -1 };
    unsigned   nExIds, i = 0;
    char CPUBrandString[0x40];
    // Get the information associated with each extended ID.
    __cpuid(CPUInfo, 0x80000000);
    nExIds = CPUInfo[0];
    for (i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(CPUInfo, i);
        // Interpret CPU brand string
        if (i == 0x80000002)
            memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000003)
            memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
        else if (i == 0x80000004)
            memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
    }
    //string includes manufacturer, model and clockspeed
    cout << "CPU Type: " << CPUBrandString << endl;
    bufferDeviceInfo << "CPU Type: " << CPUBrandString << endl;
    // Copy the hardware information to the SYSTEM_INFO structure.     
    SYSTEM_INFO siSysInfo;
    GetSystemInfo(&siSysInfo);
    printf("Number of processors: %u\n", siSysInfo.dwNumberOfProcessors);
    bufferDeviceInfo << "Number of processors: " + to_string(siSysInfo.dwNumberOfProcessors) + "\n";
    // find OpenCV build information
    //std::cout << cv::getBuildInformation() << std::endl;

    //Detect CUDA hardware
    cv::cuda::DeviceInfo deviceinfo;
    cout << "GPU: " << deviceinfo.cuda::DeviceInfo::name() << endl;
    bufferDeviceInfo << "GPU: " << deviceinfo.cuda::DeviceInfo::name() << endl;
    int cuda_devices_number = getCudaEnabledDeviceCount();
    cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
    bufferDeviceInfo << "CUDA Device(s) Number: " << cuda_devices_number << endl;
    DeviceInfo _deviceInfo;
    bool _isd_evice_compatible = _deviceInfo.isCompatible();
    cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
    bufferDeviceInfo << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

    writeTextFile("DeviceInfo.txt", bufferDeviceInfo.str());

    //Input centerPoint (has to be determined in advance) 
    cv::Point2f imCenter(2022.2, 1492.3);

    string path;
    //Input filenames of central projection and shading correction values (has to be determined in advance) and filename of images
    //cout << ">>>>>>>>> Include path to folder /Zeitmessung:" << endl;
    //cin >> path;
    string filenameImg = "../Images/1/";
    string filenameCentralProj = "../Images/corrFac.txt";
    string filenameShading = "../Images/shadingC.txt";
    string foldername = "../Images/Ergebnisse/";

    //1. Read correction values of central projection

    std::vector<double> correctionFactors = readFileDataToVector(filenameCentralProj, numImages);

    if (correctionFactors.size() == 0)
    {
        std::cout << "Could not open central projection data " << std::endl;
        return 0;
    }

    //2. Read correction values for shading correction

    std::vector<double> shadingFactors = readFileDataToVector(filenameShading, numImages);

    if (shadingFactors.size() == 0)
    {
        std::cout << "Could not open shading correction data " << std::endl;
        return 0;
    }

    // 3. read images:

    std::vector<cv::Mat> h_focusStack(numImages);
    cv::cuda::GpuMat d_refImg;
    cv::Mat refImage, h_edofImg, h_depthImg, gauss_kernelSigma4, gauss_kernelSigma8;

    cv::mulTransposed(cv::getGaussianKernel(33, 4.0, CV_32F), gauss_kernelSigma4, false);
    cv::mulTransposed(cv::getGaussianKernel(65, 8.0, CV_32F), gauss_kernelSigma8, false);

    int SobelKernel = 3;

    for (int i = 0; i < h_focusStack.size(); i++)
    {
        string wildcard;
        if (i < 10) { wildcard = "0"; }
        else { wildcard = ""; }
        string filename = filenameImg + "img" + wildcard + to_string(i) + ".png";
        h_focusStack[i] = cv::imread(filename);

        std::cout << "Image " << wildcard << to_string(i) << std::endl;

        if (h_focusStack.empty())
        {
            printf("Error opening image: %s\n", filename.c_str());
            return EXIT_FAILURE;
        }

    }

    // Due to the replicative edge treatment in central projection, edge effects occur which are eliminated by creating a reference image.
    //The reference image contains the information for each pixel which is the smallest possible depth index that the depth image may later assume.
    int numRows = h_focusStack[0].rows;
    int numCols = h_focusStack[0].cols;
    refImage = cv::Mat::zeros(numRows, numCols, CV_8UC1);
    for (int j = 0; j < idxZero; j++)
    {
        int numScaleRow = (numRows - 2 * ceil(numRows * correctionFactors[j] * 0.5)) * 0.5;
        int numScaleCol = (numCols - 2 * ceil(numCols * correctionFactors[j] * 0.5)) * 0.5;
        refImage(cv::Rect(0, 0, numScaleCol, numRows)) = cv::Scalar(j + 1);
        refImage(cv::Rect(0, 0, numCols, numScaleRow)) = cv::Scalar(j + 1);
        refImage(cv::Rect(0, numRows - numScaleRow - 1, numCols, numScaleRow)) = cv::Scalar(j + 1);
        refImage(cv::Rect(numCols - numScaleCol - 1, 0, numScaleCol, numRows)) = cv::Scalar(j + 1);
    }

    d_refImg.upload(refImage);

    cv::cuda::GpuMat d_edofImg = cv::cuda::GpuMat(h_focusStack[0].size(), CV_8UC3, Scalar(0, 0, 0));
    cv::cuda::GpuMat d_depthImg = cv::cuda::GpuMat(h_focusStack[0].size(), CV_8UC1, Scalar(0));
    cv::cuda::GpuMat d_maxSharpnessMap(h_focusStack[0].size(), CV_32FC1, Scalar(0));
    cv::cuda::GpuMat d_minSharpnessMap(h_focusStack[0].size(), CV_32FC1, Scalar(0));
    cv::cuda::GpuMat d_maskImage, d_tempImg, d_rgbImgTransformed, d_shImgs, d_idxZro_rgbImgTrnsfmd;
    for (int m = 1; m < 6; m++) {
        for (int i = 0; i < h_focusStack.size(); i++)
        {
            auto begin = std::chrono::high_resolution_clock::now();

            d_tempImg.upload(h_focusStack[i]);

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            string text = to_string(elapsed.count() * 1e-9) + " seconds";
            writeTextFile("1GPUUploadEinzelbild.txt", text);
            if (i == 1) { printf("Wall time measured, GPUUploadEinzelbild: %.3f seconds.\n", elapsed.count() * 1e-9); }

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // Performing central projection and shading correction in one function for higher performance//
            ///////////////////////////////////////////////////////////////////////////////////////////////

            begin = std::chrono::high_resolution_clock::now();

            d_rgbImgTransformed = scaleImageForCentralProjectioncAndShadingCorrection(d_tempImg, imCenter, correctionFactors[i], shadingFactors[i]);

            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds";
            writeTextFile("2GPUCentralProjectionAndShadingCorrectionEinzelbild.txt", text);
            if (i == 1) { printf("Wall time measured, GPUCentralProjectionAndShadingCorrectionEinzelbild: %.3f seconds.\n", elapsed.count() * 1e-9); }
            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Sobel Filter                                           //
            ///////////////////////////////////////////////////////////////////////////////////////////////

            begin = std::chrono::high_resolution_clock::now();

            d_shImgs = calcSharpWithSobelDeviation(d_rgbImgTransformed, SobelKernel);

            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds";
            writeTextFile("3GPUSobelEinzelbild.txt", text);
            if (i == 1) { printf("Wall time measured, GPUSobelEinzelbild: %.3f seconds.\n", elapsed.count() * 1e-9); }
            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Laplace Filter                                          //
            ///////////////////////////////////////////////////////////////////////////////////////////////
            begin = std::chrono::high_resolution_clock::now();

            d_shImgs = calcSharpWithLaplace(d_rgbImgTransformed);

            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds";
            writeTextFile("4GPULaplaceEinzelbild.txt", text);
            if (i == 1) { printf("Wall time measured, GPULaplaceEinzelbild: %.3f seconds.\n", elapsed.count() * 1e-9); }
            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Calculate Std Deviation                                 //
            ///////////////////////////////////////////////////////////////////////////////////////////////
            //////Calculate the image that shows the local sharpness

            begin = std::chrono::high_resolution_clock::now();

            d_shImgs = calculateSharpness(d_rgbImgTransformed, gauss_kernelSigma4, gauss_kernelSigma8);

            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds";
            writeTextFile("5GPUStdDevEinzelbild.txt", text);
            if (i == 1) { printf("Wall time measured, GPUStdDevEinzelbild: %.3f seconds.\n", elapsed.count() * 1e-9); }
            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Calculate EDOF-Image                                   //
            ///////////////////////////////////////////////////////////////////////////////////////////////   
            ////Calculate an image that contains the highest value of sharpness for each pixel

            begin = std::chrono::high_resolution_clock::now();

            cv::cuda::max(d_maxSharpnessMap, d_shImgs, d_maxSharpnessMap);
            cv::cuda::min(d_minSharpnessMap, d_shImgs, d_minSharpnessMap);
            cv::cuda::compare(d_maxSharpnessMap, d_shImgs, d_maskImage, cv::CMP_EQ);

            //creation Depth Image
            d_depthImg.setTo(cv::Scalar(i), d_maskImage);
            //creation EDOF Image
            d_rgbImgTransformed.copyTo(d_edofImg, d_maskImage);

            if (i == idxZero) {
                d_idxZro_rgbImgTrnsfmd = d_rgbImgTransformed;
            }

            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds";
            writeTextFile("6GPUCalculateEdofEinzelbild.txt", text);
            if (i == 1) { printf("Wall time measured, GPUCalculateEdofEinzelbild: %.3f seconds.\n", elapsed.count() * 1e-9); }
        }

        cv::cuda::GpuMat d_diffImg, d_diffImgSOBEL, d_combDiffImg, d_background, d_backgroundError;


        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                  Background Calculation                                          //
        //////////////////////////////////////////////////////////////////////////////////////////////////////

        auto begin = std::chrono::high_resolution_clock::now();

        cv::cuda::subtract(d_maxSharpnessMap, d_minSharpnessMap, d_diffImg);
        double thrsh_default = 0.007;            //0.007 StdDev
        cv::cuda::compare(d_diffImg, cv::Scalar(0.007), d_background, cv::CMP_LT);   //the choice of the Scalar is arbitrary, the quality of the background processing is adjustable

        cv::cuda::compare(d_refImg, d_depthImg, d_backgroundError, cv::CMP_GT);
        cv::cuda::bitwise_or(d_backgroundError, d_background, d_background);

        d_depthImg.setTo(cv::Scalar(idxZero), d_background);

        d_idxZro_rgbImgTrnsfmd.copyTo(d_edofImg, d_background);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        printf("Wall time measured, BackgroundCalculationEdofImage: %.3f seconds.\n", elapsed.count() * 1e-9);
        string text = to_string(elapsed.count() * 1e-9) + " seconds";
        writeTextFile("7GPUBackgroundCalculationEdofImage.txt", text);
        if (m == 1) { printf("Wall time measured, GPUBackgroundCalculationEdofImage: %.3f seconds.\n", elapsed.count() * 1e-9); }
        begin = std::chrono::high_resolution_clock::now();

        d_edofImg.download(h_edofImg);

        printf("Wall time measured, DownloadEdofImage: %.3f seconds.\n", elapsed.count() * 1e-9);
        text = to_string(elapsed.count() * 1e-9) + " seconds";
        writeTextFile("8GPUDownloadEdofImage.txt", text);
        if (m == 1) { printf("Wall time measured, GPUDownloadEdofImage: %.3f seconds.\n", elapsed.count() * 1e-9); }
        begin = std::chrono::high_resolution_clock::now();

        d_depthImg.download(h_depthImg);

        printf("Wall time measured, DownloadDepthImage: %.3f seconds.\n", elapsed.count() * 1e-9);
        text = to_string(elapsed.count() * 1e-9) + " seconds";
        writeTextFile("9GPUDownloadDepthImage.txt", text);
        if (m == 1) { printf("Wall time measured, GPUDownloadDepthImage: %.3f seconds.\n", elapsed.count() * 1e-9); }
        cout << "/////////////first of two GPU Iteration: " << to_string(m) << " of 5 /////////////" << endl;

    }
    cout << "/////////////Start second GPU Iteration for small scale measurement/////////////" << endl;
    for (int n = 1; n < 6; n++) {

        for (int i = 0; i < h_focusStack.size(); i++)
        {


            d_tempImg.upload(h_focusStack[i]);

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // Performing central projection and shading correction                                      //
            ///////////////////////////////////////////////////////////////////////////////////////////////

            d_rgbImgTransformed = scaleImageForCentralProjectioncAndShadingCorrectionSmallScale(d_tempImg, imCenter, correctionFactors[i], shadingFactors[i]);

            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Sobel Filter                                           //
            ///////////////////////////////////////////////////////////////////////////////////////////////

            d_shImgs = calcSharpWithSobelDeviationSmallScale(d_rgbImgTransformed, SobelKernel);

            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Laplace Filter                                          //
            ///////////////////////////////////////////////////////////////////////////////////////////////

            d_shImgs = calcSharpWithLaplaceSmallScale(d_rgbImgTransformed);


            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Calculate Std Deviation                                 //
            ///////////////////////////////////////////////////////////////////////////////////////////////
            //////Calculate the image that shows the local sharpness


            d_shImgs = calculateSharpnessSmallScale(d_rgbImgTransformed, gauss_kernelSigma4, gauss_kernelSigma8);

            ////////////////////////////////////////////////////////////////////////////////////////////////
            //                                    Calculate EDOF-Image                                   //
            ///////////////////////////////////////////////////////////////////////////////////////////////   
            ////Calculate an image that contains the highest value of sharpness for each pixel

            auto begin = std::chrono::high_resolution_clock::now();
            cv::cuda::max(d_maxSharpnessMap, d_shImgs, d_maxSharpnessMap);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            string text = to_string(elapsed.count() * 1e-9) + " seconds 35WallTimeMax(d_maxSharpnessMap, d_shImgs, d_maxSharpnessMap)";
            writeTextFile("14GPUCalculateEdofEinzelbildSmallScale.txt", text);

            begin = std::chrono::high_resolution_clock::now();
            cv::cuda::min(d_minSharpnessMap, d_shImgs, d_minSharpnessMap);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 36WallTimeMin(d_minSharpnessMap, d_shImgs, d_minSharpnessMap)";
            writeTextFile("15GPUCalculateEdofEinzelbildSmallScale.txt", text);

            begin = std::chrono::high_resolution_clock::now();
            cv::cuda::compare(d_maxSharpnessMap, d_shImgs, d_maskImage, cv::CMP_EQ);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 37WallTimeCompare(d_maxSharpnessMap, d_shImgs, d_maskImage, CMP_EQ)";
            writeTextFile("16GPUCalculateEdofEinzelbildSmallScale.txt", text);

            //creation Depth Image
            begin = std::chrono::high_resolution_clock::now();
            d_depthImg.setTo(cv::Scalar(i), d_maskImage);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 38WallTimeD_depthImg.setTo(cv::Scalar(i), d_maskImage)";
            writeTextFile("17GPUCalculateEdofEinzelbildSmallScale.txt", text);

            //creation EDOF Image
            begin = std::chrono::high_resolution_clock::now();
            d_rgbImgTransformed.copyTo(d_edofImg, d_maskImage);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 39WallTimeD_rgbImgTransformed.copyTo(d_edofImg, d_maskImage)";
            writeTextFile("18GPUCalculateEdofEinzelbildSmallScale.txt", text);


            if (i == idxZero) {
                d_idxZro_rgbImgTrnsfmd = d_rgbImgTransformed;
            }

        }

        cv::cuda::GpuMat d_diffImg, d_diffImgSOBEL, d_combDiffImg, d_background, d_backgroundError;

        auto begin = std::chrono::high_resolution_clock::now();
        cv::cuda::subtract(d_maxSharpnessMap, d_minSharpnessMap, d_diffImg);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        string text = to_string(elapsed.count() * 1e-9) + " seconds 40WallTimeSubtract(d_maxSharpnessMap, d_minSharpnessMap, d_diffImg)";
        writeTextFile("19GPUCalculateEdofEinzelbildSmallScale.txt", text);
        //////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                  Background Calculation                                          //
        //////////////////////////////////////////////////////////////////////////////////////////////////////

        double thrsh_default = 0.007;            //0.007 StdDev

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::compare(d_diffImg, cv::Scalar(0.007), d_background, cv::CMP_LT);   //the choice of the Scalar is arbitrary, the quality of the background processing is adjustable
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 41WallTimeCompare(d_diffImg, Scalar(0.007), d_background, CMP_LT)";
        writeTextFile("20GPUBackgroundCalculationEdofImageSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::compare(d_refImg, d_depthImg, d_backgroundError, cv::CMP_GT);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 42WallTimeCompare(d_refImg, d_depthImg, d_backgroundError, CMP_GT)";
        writeTextFile("21GPUBackgroundCalculationEdofImageSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::cuda::bitwise_or(d_backgroundError, d_background, d_background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 43WallTimeBitwise_or(d_backgroundError, d_background, d_background)";
        writeTextFile("22GPUBackgroundCalculationEdofImageSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        d_depthImg.setTo(cv::Scalar(idxZero), d_background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 44WallTimeD_depthImg.setTo(Scalar(idxZero), d_background)";
        writeTextFile("23GPUBackgroundCalculationEdofImageSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        d_idxZro_rgbImgTrnsfmd.copyTo(d_edofImg, d_background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 45WallTimeD_idxZro_rgbImgTrnsfmd.copyTo(d_edofImg, d_background)";
        writeTextFile("24GPUBackgroundCalculationEdofImageSmallScale.txt", text);

        cout << "/////////////second GPU Iteration for small scale measurement: " << to_string(n) << " of 5 /////////////" << endl;
    }

    cout << "///////////// Start first of two CPU Run /////////////" << endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //               Performing central projection and shading correction                        //
    ///////////////////////////////////////////////////////////////////////////////////////////////

    //1. central projection

    std::vector<cv::Mat> rgbImgsTransformed(numImages);
    for (int p = 1; p < 6; p++) {
        auto begin = std::chrono::high_resolution_clock::now();

        concurrency::parallel_transform(h_focusStack.begin(), h_focusStack.end(), correctionFactors.begin(), rgbImgsTransformed.begin(),
            [imCenter](cv::Mat& img, double corrVal) { return scaleImageForCentralProjectionc(img, imCenter, corrVal); });

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("Wall time measured, CPU CentralProjection: %.3f seconds.\n", elapsed.count() * 1e-9);
        string text = to_string(elapsed.count() * 1e-9) + " seconds";
        writeTextFile("25CPUCentralProjectionConcurrencyFocusStack.txt", text);

        //std::cout << "Images are scaled" << std::endl;



        //2. Shading correction

        begin = std::chrono::high_resolution_clock::now();

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shadingFactors.begin(), rgbImgsTransformed.begin(),
            [](cv::Mat& img, double sc) { return shadingCorrection(img, sc); });

        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("Wall time measured, CPU Shading correction: %.3f seconds.\n", elapsed.count() * 1e-9);
        text = to_string(elapsed.count() * 1e-9) + " seconds";
        writeTextFile("26CPUShadingCorrectionConcurrencyFocusStack.txt", text);

        ////////////////////////////////////////////////////////////////////////////////////////////////
        //                                    Calculate EDOF-Image                                   //
        ///////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////
       //                                    Laplace Filter                                          //
       ///////////////////////////////////////////////////////////////////////////////////////////////
        ////Calculate the image that shows the local sharpness
        std::vector<cv::Mat> shImgs(rgbImgsTransformed.size());

        auto begin_2 = std::chrono::high_resolution_clock::now();

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shImgs.begin(),
            [](cv::Mat& img) {return calcSharpWithLaplace(img); });

        auto end_2 = std::chrono::high_resolution_clock::now();
        auto elapsed_2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_2 - begin_2);
        printf("Wall time measured, Laplace Filter: %.3f seconds.\n", elapsed_2.count() * 1e-9);
        text = to_string(elapsed_2.count() * 1e-9) + " seconds";
        writeTextFile("27CPULaplaceConcurrencyFocusStack.txt", text);
        ////////////////////////////////////////////////////////////////////////////////////////////////
        //                                    Sobel Filter                                           //
        ///////////////////////////////////////////////////////////////////////////////////////////////

        begin_2 = std::chrono::high_resolution_clock::now();

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shImgs.begin(),
            [](cv::Mat& img) {return calcSharpWithSobelDeviation(img); });

        end_2 = std::chrono::high_resolution_clock::now();
        elapsed_2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_2 - begin_2);
        printf("Wall time measured, Sobel Filter: %.3f seconds.\n", elapsed_2.count() * 1e-9);
        text = to_string(elapsed_2.count() * 1e-9) + " seconds";
        writeTextFile("28CPUSobelConcurrencyFocusStack.txt", text);

        ////////////////////////////////////////////////////////////////////////////////////////////////
        //                                  Std Dev                                                  //
       ///////////////////////////////////////////////////////////////////////////////////////////////
        begin = std::chrono::high_resolution_clock::now();

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shImgs.begin(),
            [](cv::Mat& img) {return calculateSharpness(img); });

        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("Wall time measured, Standard deviation: %.3f seconds.\n", elapsed.count() * 1e-9);
        text = to_string(elapsed.count() * 1e-9) + " seconds";
        writeTextFile("29CPUStdDevConcurrencyFocusStack.txt", text);

        ///////////////////////////////////////////////////////////////////////////////////////////////       
         ///////////////////////////////////////////////////////////////////////////////////////////////    
         ////Calculate an image that contains the highest value of sharpness for each pixel



        cv::Mat maxImg = shImgs[0].clone();

        begin = std::chrono::high_resolution_clock::now();
        concurrency::parallel_for_each(shImgs.begin() + 1, shImgs.end(), [&maxImg](cv::Mat img) { cv::max(maxImg, img, maxImg); });
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 51CPUMaxParallel_for_eachFocusStack";
        writeTextFile("30CPUCalculateEdofFocusStackSmallScale.txt", text);

        //Calculate an image that contains the lowest value of sharpness for each pixel (probably not necessary)
        cv::Mat minImg = shImgs[0].clone();

        begin = std::chrono::high_resolution_clock::now();
        concurrency::parallel_for_each(shImgs.begin() + 1, shImgs.end(), [&minImg](cv::Mat img) { cv::min(minImg, img, minImg); });
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 52CPUMinParallel_for_eachConcurrencyFocusStack";
        writeTextFile("31CPUCalculateEdofFocusStackSmallScale.txt", text);

        //Calculate an image that contains the background (background is the part of the image, where the leven of sharpness is very low)
        begin = std::chrono::high_resolution_clock::now();
        cv::Mat diffImg = maxImg - minImg;
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 53CPUdiffImg = maxImg - minImg";
        writeTextFile("32CPUCalculateEdofFocusStackSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::Mat background = diffImg < 0.01;
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 54CPUbackground = diffImg  0.01";
        writeTextFile("33CPUCalculateEdofFocusStackSmallScale.txt", text);
        //Calculate EDOF and depth image
        cv::Mat edofImg = cv::Mat::zeros(rgbImgsTransformed[0].size(), rgbImgsTransformed[0].type());
        cv::Mat depthImg = cv::Mat::zeros(maxImg.size(), CV_8UC1);

        for (int i = 0; i < numImages; i++)
        {
            begin = std::chrono::high_resolution_clock::now();
            cv::Mat maskImage = getSharpestPixel(maxImg, shImgs[i], rgbImgsTransformed[i], edofImg);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 55CPUGetSharpestPixel";
            writeTextFile("34CPUCalculateEdofFocusStackSmallScale.txt", text);

            begin = std::chrono::high_resolution_clock::now();
            cv::subtract(depthImg, cv::Scalar(255), depthImg, maskImage);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 56CPUSubtract(depthImg, Scalar(255), depthImg, maskImage)";
            writeTextFile("35CPUCalculateEdofFocusStackSmallScale.txt", text);

            begin = std::chrono::high_resolution_clock::now();
            cv::add(depthImg, cv::Scalar(i), depthImg, maskImage);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 57CPUAdd(depthImg, Scalar(i), depthImg, maskImage)";
            writeTextFile("36CPUCalculateEdofFocusStackSmallScale.txt", text);
            //
            begin = std::chrono::high_resolution_clock::now();
            cv::subtract(edofImg, cv::Scalar(255, 255, 255), edofImg, maskImage);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 58CPUSubtract(edofImg, Scalar(255, 255, 255), edofImg, maskImage)";
            writeTextFile("37CPUCalculateEdofFocusStackSmallScale.txt", text);

            begin = std::chrono::high_resolution_clock::now();
            cv::add(edofImg, rgbImgsTransformed[i], edofImg, maskImage);
            end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            text = to_string(elapsed.count() * 1e-9) + " seconds 59CPUAdd(edofImg, rgbImgsTransformed, edofImg, maskImage)";
            writeTextFile("38CPUCalculateEdofFocusStackSmallScale.txt", text);
        }



        //set background to values of images at F=0 dpt

        //The background is on the one hand the area which contains errors due to the boundary treatment


        cv::Mat backgroundError;
        begin = std::chrono::high_resolution_clock::now();
        cv::compare(refImage, depthImg, backgroundError, cv::CMP_GT);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 60CPUcompare(refImage, depthImg, backgroundError, CMP_GT)";
        writeTextFile("39CPUBackgroundCalculationEdofImageSmallScale.txt", text);

        //And background is the area where the sharpness is very low
        begin = std::chrono::high_resolution_clock::now();
        cv::bitwise_or(backgroundError, background, background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 61CPUbitwise_or(backgroundError, background, background)";
        writeTextFile("40CPUBackgroundCalculationEdofImageSmallScale.txt", text);

        begin = std::chrono::high_resolution_clock::now();
        cv::subtract(depthImg, cv::Scalar(255), depthImg, background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 62CPUsubtract(depthImg, Scalar(255), depthImg, background)";
        writeTextFile("41CPUBackgroundCalculationEdofImageSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        cv::add(depthImg, cv::Scalar(idxZero), depthImg, background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 63CPUadd(depthImg, Scalar(idxZero), depthImg, background)";
        writeTextFile("42CPUBackgroundCalculationEdofImageSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        cv::subtract(edofImg, cv::Scalar(255, 255, 255), edofImg, background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 64CPUsubtract(edofImg, Scalar(255, 255, 255), edofImg, background)";
        writeTextFile("43CPUBackgroundCalculationEdofImageSmallScale.txt", text);
        begin = std::chrono::high_resolution_clock::now();
        cv::add(edofImg, rgbImgsTransformed[idxZero], edofImg, background);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        text = to_string(elapsed.count() * 1e-9) + " seconds 65CPUadd(edofImg, rgbImgsTransformed[idxZero], edofImg, background)";
        writeTextFile("44CPUBackgroundCalculationEdofImageSmallScale.txt", text);

        cout << "/////////////first of two CPU Iteration: " << to_string(p) << " of 5 /////////////" << endl;
    }
    cout << "/////////////Start second CPU Iteration /////////////" << endl;
    for (int r = 1; r < 6; r++) {


        concurrency::parallel_transform(h_focusStack.begin(), h_focusStack.end(), correctionFactors.begin(), rgbImgsTransformed.begin(),
            [imCenter](cv::Mat& img, double corrVal) { return scaleImageForCentralProjectioncSmallScale(img, imCenter, corrVal); });


        //std::cout << "Images are scaled" << std::endl;



        //2. Shading correction

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shadingFactors.begin(), rgbImgsTransformed.begin(),
            [](cv::Mat& img, double sc) { return shadingCorrectionSmallScale(img, sc); });



        ////////////////////////////////////////////////////////////////////////////////////////////////
        //                                    Calculate EDOF-Image                                   //
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ////Calculate the image that shows the local sharpness
        std::vector<cv::Mat> shImgs(rgbImgsTransformed.size());

        ////////////////////////////////////////////////////////////////////////////////////////////////
        //                                    Sobel Filter                                           //
        ///////////////////////////////////////////////////////////////////////////////////////////////

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shImgs.begin(),
            [](cv::Mat& img) {return calcSharpWithSobelDeviationSmallScale(img); });
        ////////////////////////////////////////////////////////////////////////////////////////////////
   //                                    Laplace Filter                                          //
   ///////////////////////////////////////////////////////////////////////////////////////////////


        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shImgs.begin(),
            [](cv::Mat& img) {return calcSharpWithLaplaceSmallScale(img); });

        ////////////////////////////////////////////////////////////////////////////////////////////////
        //                                  Std Dev                                                  //
       ///////////////////////////////////////////////////////////////////////////////////////////////

        concurrency::parallel_transform(rgbImgsTransformed.begin(), rgbImgsTransformed.end(), shImgs.begin(),
            [](cv::Mat& img) {return calculateSharpnessSmallScale(img); });

        cout << "/////////////second CPU Iteration: " << to_string(r) << " of 5 /////////////" << endl;
    }
    cout << "////////////////////////thank you ////////////////////////" << endl;
    cout << "please send me the .txt files from the folder Zeitmessung_EDOF_CPU_GPU, to kimohnesorg@hotmail.de" << endl;
    cout << "Exit with any key and enter" << endl;
    cin >> path;
    return 0;

}