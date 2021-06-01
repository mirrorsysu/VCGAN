source activate slowmo
python train.py \
--stage 'First' \
--log_path './log.csv' \
--load_model True \
--load_path './models/First_Stage_epoch50_bs56.pth' \
--load_fe True \
--fe_path './trained_models/resnet50_in_rgb_epoch150_bs512.pth' \
--load_perceptual True \
--perceptual_path 'trained_models/vgg16-397923af.pth' \
--begin_epoch 51 \
\
--save_by_epoch 1 \
--save_path './models' \
--sample_path './samples' \
\
--multi_gpu True \
--cudnn_benchmark True \
\
--pwcnet_path './trained_models/pwcNet-default.pytorch' \
--video_class_txt './txt/DAVIS_videvo_train_class.txt' \
--video_imagelist_txt './txt/DAVIS_videvo_train_imagelist.txt' \
\
--epochs 101 \
--batch_size 56 \
--lr_g 1e-4 \
--lr_d 1e-4 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--gan_mode 'no' \
--lambda_l1 10 \
--lambda_percep 5 \
--lambda_gan 1 \
--lambda_flow 0 \
--lambda_flow_short 3 \
--lambda_flow_long 5 \
--mask_para 50 \
--lambda_gp 10 \
--pad 'reflect' \
--activ_g 'lrelu' \
--activ_d 'lrelu' \
--norm 'in' \
--init_type 'xavier' \
--init_gain 0.02 \
--baseroot '../../DATASETS/ILSVRC-256/' \
--testroot '../../DATASETS/ILSVRC/Data/DET/test/' \
--sample_size 1 \
--crop_size 256 \
--crop_size_h 256 \
--crop_size_w 448 \
--geometry_aug False \
--angle_aug False \
--scale_min 1 \
--scale_max 1 \
