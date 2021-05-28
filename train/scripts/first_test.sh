python -m tester.test_model_first_stage_by_folder \
--folderpath '../../DATASETS/ILSVRC/Data/DET/test/' \
--savepath './first_stage_results' \
--load_name './models/First_Stage_epoch67_bs64.pth' \
--test_gpu 0