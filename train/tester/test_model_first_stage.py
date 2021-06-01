# Test the model's first stage
import os
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
from networks import network
from pathlib import Path

def test(grayimg, model):
    # Forward and reshape to [H, W, C], in range [-1, 1]
    out_rgb = model(grayimg)
    out_rgb = out_rgb.squeeze(0).cpu().detach().numpy().reshape([3, 256, 256])
    out_rgb = out_rgb.transpose(1, 2, 0)
    # Return the original scale
    out_rgb = (out_rgb * 0.5 + 0.5) * 255
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb


def getImage(imgpath):
    # Read the images
    grayimg = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    if grayimg.shape[:2] != (256, 256):
        grayimg = cv2.resize(grayimg, (256, 256))
    grayimg = np.expand_dims(grayimg, 2)
    grayimg = np.concatenate((grayimg, grayimg, grayimg), axis=2)
    # Normalized to [-1, 1]
    grayimg = (grayimg.astype(np.float64) - 128) / 128
    # To PyTorch Tensor
    grayimg = torch.from_numpy(grayimg.transpose(2, 0, 1).astype(np.float32)).contiguous()
    grayimg = grayimg.unsqueeze(0).to(opt.device)
    return grayimg


def load_model(opt, index):
    model = network.FirstStageNet(opt)
    load_path = opt.load_name
    if '%' in load_path:
        load_path = load_path % (index)
    print('load:', load_path)
    pretrained_dict = torch.load(load_path)

    # Get the dict from processing network
    process_dict = model.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    model.load_state_dict(process_dict)
    model = model.to(opt.device)
    return model, load_path

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # testing parameters
    parser.add_argument(
        "--imgpath",
        type=str,
        default="./first_stage_results/ILSVRC2012_test_00000358_gt.JPEG",
        help="testing image path",
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="./first_stage_all_results",
        help="saving folder path",
    )
    parser.add_argument("--load_name", type=str, default="./models/First_Stage_epoch%d_bs56.pth", help="load the pre-trained model with certain epoch")
    parser.add_argument("--start", type=int, default=0, help="the start of test model")
    parser.add_argument("--end", type=int, default=0, help="the end of test model")

    parser.add_argument("--crop_size", type=int, default=256, help="single patch size")
    parser.add_argument("--comparison", type=bool, default=False, help="compare with original RGB image or not")
    # GPU parameters
    parser.add_argument("--test_gpu", type=str, default="-1", help="gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU")
    # network parameters
    parser.add_argument("--in_channels", type=int, default=1, help="in channel for U-Net encoder")
    parser.add_argument("--start_channels", type=int, default=32, help="start channel for U-Net encoder")
    parser.add_argument("--out_channels", type=int, default=3, help="out channel for U-Net decoder")
    parser.add_argument("--pad", type=str, default="reflect", help="padding type")
    parser.add_argument("--activ_g", type=str, default="lrelu", help="activation function for generator")
    parser.add_argument("--activ_d", type=str, default="lrelu", help="activation function for discriminator")
    parser.add_argument("--norm", type=str, default="in", help="normalization type")
    opt = parser.parse_args()
    if opt.test_gpu == "-1":
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device("cuda:{}".format(opt.test_gpu))
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.test_gpu
    print("Single-GPU mode, %s GPU is used" % (opt.test_gpu))

    # ----------------------------------------
    #                  Testing
    # ----------------------------------------
    check_path(opt.savepath)

    # Get image
    img = getImage(opt.imgpath)

    for index in range(opt.start, opt.end + 1):
        # Get model
        model, load_path = load_model(opt, index)

        # Get result [H, W, C], in range [0, 255]
        out_rgb = test(img, model)
        out_rgb = out_rgb[:, :, ::-1]

        # Print
        savepath_pre = os.path.join(opt.savepath, '{}.jpg'.format(Path(load_path).stem))
        cv2.imwrite(savepath_pre, out_rgb)
