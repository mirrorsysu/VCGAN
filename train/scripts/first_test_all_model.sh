source activate slowmo
python -m tester.test_model_first_stage \
--imgpath '../../DATASETS/ILSVRC/Data/DET/test/ILSVRC2012_test_00000978.JPEG' \
--savepath './first_stage_all_results' \
--load_name './models/First_Stage_epoch%d_bs56.pth' \
--start 0 \
--end 51 \
--test_gpu -1