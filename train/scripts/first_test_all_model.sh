source activate slowmo
python -m tester.test_model_first_stage \
--imgpath '/home/data/mir-workspace/VCGAN/train/first_stage_results/ILSVRC2016_test_00007977_gt.JPEG' \
--savepath './first_stage_all_results' \
--load_name './models/First_Stage_epoch%d_bs56.pth' \
--start 0 \
--end 100 \
--test_gpu -1