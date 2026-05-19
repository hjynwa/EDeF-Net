#!/bin/bash
# file=('01' '02' '03' '04' '05' '11') # 20230505
# file=('01' '05' '06' '07'  '09') # 20230509
# file=('20240201_001_mirror' '20240201_022_mirror' '20240201_025_mirror' '20240203_000_mirror' '20240203_001_mirror' '20240203_003_mirror' '20240203_007_mirror' '20240203_009_mirror' '20240203_010_mirror' '20240203_012_mirror' '20240203_013_mirror' '20240203_015_mirror' '20240203_025_mirror' '20240204_004_mirror' '20240204_005_mirror' '20240204_011_mirror' '20240204_012_mirror' '20240204_021_mirror' '20240204_023_mirror' '20240204_024_mirror' '20240204_025_mirror' '20240204_026_mirror' '20240204_030_mirror') # bothview
# file=('20240203_000_mirror' '20240203_007_mirror' '20240204_025_mirror' '20240204_026_mirror')
# for f in ${file[@]};
# do
#     # python infer.py --config saved/models/TASA_nonormloss_midres_olddata/0304_010857/config.json --resume saved/models/TASA_nonormloss_midres_olddata/0304_010857/checkpoint-epoch20.pth  --run_id 0304_010857 --data_dir ../NeurIPS23_code/data/real_data/bothview/$f --num_test 800
#     python scripts/merge_result_stacks.py --input_dir saved/results/TASA_nonormloss_midres_olddata/0304_010857/bothview/$f --interval 10000 --real

#     echo "$f finished"
# done

# file2=('01' '02' '03' '04' '05' '06' '07' '08' '09' '10') # 20230509
# file2=('01' '02' '03') # 20230509
# for f in ${file2[@]};
# do
#     python infer.py --config saved/models/EDeFT_3dconv_tatimes_full/0509_184033/config.json --resume saved/models/EDeFT_3dconv_tatimes_full/0509_184033/model_best.pth  --run_id 0509_184033 --data_dir data/real_data/20230523_track/$f
#     python scripts/merge_result_stacks.py --input_dir saved/results/EDeFT_3dconv_tatimes_full/0509_184033/20230523_track/$f --interval 10000 --real

#     echo "$f finished"
# done

file2=('flicker_000' 'flicker_001' 'flicker_004' 'flicker_006' 'flicker_008' 'flicker_021' 'flicker_027' 'flicker_028' 'flicker_032' 'flicker_037' 'flicker_038' 'flicker_101' 'flicker_103' 'flicker_104' 'flicker_105' 'flicker_106' 'flicker_107' 'flicker_108' 'flicker_109' 'flicker_110') 

for f in ${file2[@]};
do
    # python infer.py --config saved/models/TASA_nonormloss_midres_olddata/0304_010857/config.json --resume saved/models/TASA_nonormloss_midres_olddata/0304_010857/checkpoint-epoch60.pth  --run_id 0304_010857 --data_dir /media/hanjin/4T_HDD/Ubuntu/Workspace/Event_flicker/NeurIPS23_code/data/synthetic_tracking/$f
    # python scripts/merge_result_stacks.py --input_dir saved/results/TASA_nonormloss_midres_olddata/0304_010857/synthetic_tracking/$f --interval 10000 --real
    cp saved/results/TASA_nonormloss_midres_olddata/0304_010857/synthetic_tracking/$f/evs_txt_merged/*.txt /media/hanjin/4T_HDD/Ubuntu/Github_repos/CVPR2022_STNet/tracking_data/evs_txt/ours_ECCV24/

    echo "$f finished"
done

# file2=('flicker_000' 'flicker_001' 'flicker_004' 'flicker_006' 'flicker_008' 'flicker_021' 'flicker_027' 'flicker_028' 'flicker_032' 'flicker_037' 'flicker_038' 'flicker_101' 'flicker_103' 'flicker_104' 'flicker_105' 'flicker_106' 'flicker_107' 'flicker_108' 'flicker_109' 'flicker_110') 

# for f in ${file2[@]};
# do
#     python test.py --config saved/models/NM_ablations/Ours_N6M4/0509_184033/config.json --resume saved/models/NM_ablations/Ours_N6M4/0509_184033/checkpoint-epoch150.pth  --run_id 0509_184033

#     echo "$f finished"
# done