import argparse
import numpy as np
import os
import cv2
from skimage import io
from skimage import measure
from skimage import transform
from skimage import color

# Compute the mean-squared error between two images
def MSE(srcpath, dstpath, gray2rgb=False, scale=256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale != (0, 0):
        scr = cv2.resize(scr, scale)
        dst = cv2.resize(dst, scale)
    mse = measure.compare_mse(scr, dst)
    return mse


# Compute the normalized root mean-squared error (NRMSE) between two images
def NRMSE(srcpath, dstpath, gray2rgb=False, scale=256, mse_type="Euclidean"):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale != (0, 0):
        scr = cv2.resize(scr, scale)
        dst = cv2.resize(dst, scale)
    nrmse = measure.compare_nrmse(scr, dst, norm_type=mse_type)
    return nrmse


# Compute the peak signal to noise ratio (PSNR) for an image
def PSNR(srcpath, dstpath, gray2rgb=False, scale=256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale != (0, 0):
        scr = cv2.resize(scr, scale)
        dst = cv2.resize(dst, scale)
    psnr = measure.compare_psnr(scr, dst)
    return psnr


# Compute the mean structural similarity index between two images
def SSIM(srcpath, dstpath, gray2rgb=False, scale=256, RGBinput=True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale != (0, 0):
        scr = cv2.resize(scr, scale)
        dst = cv2.resize(dst, scale)
    ssim = measure.compare_ssim(scr, dst, multichannel=RGBinput)
    return ssim


# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, "r")
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][: len(content[i]) - 1]
    file.close()
    return content


# save a list to a txt
def text_save(content, filename, mode="a"):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + "\n")
    file.close()


# Traditional indexes accuracy for dataset
def Dset_Acuuracy(imglist, refpath, basepath, gray2rgb=False, scale=(0, 0)):
    # Define the list saving the accuracy
    nrmselist = []
    psnrlist = []
    ssimlist = []
    nrmseratio = 0
    psnrratio = 0
    ssimratio = 0

    # Compute the accuracy
    for i in range(len(imglist)):
        # Full imgpath
        imgname = imglist[i]  # + '.JPEG'
        refimgpath = os.path.join(refpath, imgname)
        imgpath = os.path.join(basepath, imgname)
        # Compute the traditional indexes
        nrmse = NRMSE(refimgpath, imgpath, gray2rgb, scale, "Euclidean")
        psnr = PSNR(refimgpath, imgpath, gray2rgb, scale)
        ssim = SSIM(refimgpath, imgpath, gray2rgb, scale, True)
        nrmselist.append(nrmse)
        psnrlist.append(psnr)
        ssimlist.append(ssim)
        nrmseratio = nrmseratio + nrmse
        psnrratio = psnrratio + psnr
        ssimratio = ssimratio + ssim
        print("The %dth image: nrmse: %f, psnr: %f, ssim: %f" % (i, nrmse, psnr, ssim))
    nrmseratio = nrmseratio / len(imglist)
    psnrratio = psnrratio / len(imglist)
    ssimratio = ssimratio / len(imglist)

    return nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--imglist", type=str, default="./DAVIS_test_imagelist_without_first_frame.txt", help="define imglist txt path")
    parser.add_argument("--refpath", type=str, default="", help="define reference path")
    parser.add_argument("--basepath", type=str, default="", help="define imgpath")
    parser.add_argument("--gray2rgb", type=bool, default=False, help="whether there is an input is grayscale")
    parser.add_argument("--scale", type=tuple, default=(256, 256), help="whether the input needs resize")
    parser.add_argument("--savelist", type=bool, default=False, help="whether the results should be saved")
    opt = parser.parse_args()
    print(opt)

    # Read all names
    imglist = text_readlines(opt.imglist)

    nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio = Dset_Acuuracy(imglist, opt.refpath, opt.basepath, gray2rgb=opt.gray2rgb, scale=opt.scale)

    print("The overall results: nrmse: %f, psnr: %f, ssim: %f" % (nrmseratio, psnrratio, ssimratio))

    # Save the files
    if opt.savelist:
        text_save(nrmselist, "./nrmselist.txt")
        text_save(psnrlist, "./psnrlist.txt")
        text_save(ssimlist, "./ssimlist.txt")
