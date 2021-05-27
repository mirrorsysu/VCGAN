import argparse
from pathlib import Path


def str2bool(str):
    return True if str.lower() == "true" else False


def str2Path(str):
    return Path(str)


if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # pre-train, saving, and loading parameters
    parser.add_argument("--stage", type=str, help="First or Second")
    parser.add_argument("--load_model", type=str2bool, default=False, help="load model ot not")
    parser.add_argument("--load_path", type=str2Path, default="./models/First_Stage_final.pth", help="load the pre-trained model with certain epoch")
    parser.add_argument("--load_fe", type=str2bool, default=True, help="load pretrain fe or not")
    parser.add_argument("--fe_path", type=str2Path, default="./trained_models/resnet50_in_epoch150_bs256.pth", help="the path that contains the pre-trained ResNet model")
    parser.add_argument("--load_perceptual", type=str2bool, default=True, help="load pretrain perceptual or not")
    parser.add_argument("--perceptual_path", type=str2Path, default="./trained_models/vgg16_pretrained.pth", help="the path that contains the pre-trained VGG-16 model")
    parser.add_argument("--begin_epoch", type=int, default=0, help="the begin epoch of train")

    parser.add_argument("--save_by_epoch", type=int, default=1, help="interval between model checkpoints (by epochs)")
    parser.add_argument("--save_path", type=str2Path, default="./models", help="save the pre-trained model to certain path")
    parser.add_argument("--sample_path", type=str2Path, default="./samples", help="save the sample to certain path")

    # GPU parameters
    parser.add_argument("--multi_gpu", type=str2bool, default=False, help="True for more than 1 GPU, we recommend to use 4 NVIDIA Tesla v100 GPUs")
    parser.add_argument("--cudnn_benchmark", type=str2bool, default=True, help="True for unchanged input data type")

    parser.add_argument("--pwcnet_path", type=str2Path, default="./trained_models/pwcNet-default.pytorch", help="the path that contains the pre-trained PWCNet model")
    parser.add_argument("--video_class_txt", type=str2Path, default="./txt/DAVIS_videvo_train_class.txt", help="the path that contains DAVIS_videvo_train_class.txt")
    parser.add_argument("--video_imagelist_txt", type=str2Path, default="./txt/DAVIS_videvo_train_imagelist.txt", help="the path that contains DAVIS_videvo_train_imagelist.txt")

    # dataset
    parser.add_argument("--baseroot", type=str2Path, default="C:\\Users\\yzzha\\Desktop\\dataset\\1", help="color image baseroot")
    parser.add_argument("--num_workers", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--sample_size", type=int, default=1, help="sample number for the dataset at first stage")
    parser.add_argument("--crop_size", type=int, default=256, help="single patch size")  # first stage: 256 * 256
    parser.add_argument("--crop_size_h", type=int, default=256, help="single patch size")  # second stage (128p, 256p, 448p): 128, 256, 448
    parser.add_argument("--crop_size_w", type=int, default=448, help="single patch size")  # second stage (128p, 256p, 448p): 256, 448, 832
    parser.add_argument("--geometry_aug", type=str2bool, default=False, help="geometry augmentation (scaling)")
    parser.add_argument("--angle_aug", type=str2bool, default=False, help="geometry augmentation (rotation, flipping)")
    parser.add_argument("--scale_min", type=float, default=1, help="min scaling factor")
    parser.add_argument("--scale_max", type=float, default=1, help="max scaling factor")

    # training parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs of training")  # change if fine-tune
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Adam: learning rate for G")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Adam: learning rate for D")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second order momentum of gradient")
    parser.add_argument("--weight_decay", type=int, default=0, help="weight decay for optimizer")
    parser.add_argument("--lr_decrease_epoch", type=int, default=10, help="lr decrease at certain epoch and its multiple")
    parser.add_argument("--lr_decrease_factor", type=float, default=0.5, help="lr decrease factor")
    parser.add_argument("--gan_mode", type=str, default="no", help="type of GAN: [no | LSGAN | WGAN | WGANGP], WGAN is recommended")

    # loss balancing parameters
    parser.add_argument("--lambda_l1", type=float, default=10, help="coefficient for L1 Loss")
    parser.add_argument("--lambda_percep", type=float, default=5, help="coefficient for Perceptual Loss")
    parser.add_argument("--lambda_gan", type=float, default=1, help="coefficient for GAN Loss")
    parser.add_argument("--lambda_flow", type=float, default=0, help="coefficient for Flow Loss")
    parser.add_argument("--lambda_flow_short", type=float, default=3, help="coefficient for Perceptual Loss")
    parser.add_argument("--lambda_flow_long", type=float, default=3, help="coefficient for Perceptual Loss")
    parser.add_argument("--mask_para", type=float, default=50, help="coefficient for visible mask")
    parser.add_argument("--lambda_gp", type=float, default=10, help="coefficient for WGAN-GP coefficient")

    # network parameters
    parser.add_argument("--in_channels", type=int, default=1, help="in channel for U-Net encoder")
    parser.add_argument("--start_channels", type=int, default=32, help="start channel for U-Net encoder")
    parser.add_argument("--out_channels", type=int, default=3, help="out channel for U-Net decoder")
    parser.add_argument("--pad", type=str, default="reflect", help="padding type")
    parser.add_argument("--activ_g", type=str, default="lrelu", help="activation function for generator")
    parser.add_argument("--activ_d", type=str, default="lrelu", help="activation function for discriminator")
    parser.add_argument("--norm", type=str, default="in", help="normalization type")
    parser.add_argument("--init_type", type=str, default="xavier", help="intialization type for generator and discriminator")
    parser.add_argument("--init_gain", type=float, default=0.02, help="the standard deviation if Gaussian normalization")

    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    from trainer import trainer

    if opt.gan_mode == "no":
        trainer.trainer_noGAN(opt)
    if opt.gan_mode == "LSGAN":
        trainer.trainer_LSGAN(opt)
    if opt.gan_mode == "WGAN":
        trainer.trainer_WGAN(opt)
    if opt.gan_mode == "WGANGP":
        trainer.trainer_WGANGP(opt)
