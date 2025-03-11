import os

GPU_ID = 3

############# train参数设置 #############
# python train.py -s [path_to_dataset] --sparse_num [N]( 1 ==> 100% images, 5 ==> 20% images)
# --sparse_num 只有在从KeyFrameTrajectory2.txt文件中创建相机才起作用，--eval默认开启
# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python train.py \
#         -s /data2/liuzhi/Dataset/3DGS_Dataset/input/gm_Museum \
#         -m ./output/gm_Museum \
#         --sparse_num 1'
# print(cmd)
# os.system(cmd)


# print("----------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python render.py \
#         -m ./output/Replica-OFF2'
# print(cmd)
# os.system(cmd)


# scenes = {"building1": "cuda", "building2": "cpu", "building3": "cpu", "town1": "cuda", "urban/block2_sxfx": "cpu"}
scenes = {"building1": "cuda"}
for idx, scene in enumerate(scenes.items()):
    ############ 训练 ############
    # print("--------------------------------------------------------------")
    # cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
    #         python train.py \
    #         -s ../../remote_data/dataset_simulator/{scene[0]}/train \
    #         -m output_gt_traingulate_1600/{scene[0]} \
    #         -r -1 \
    #         --sparse_num 1 \
    #         --data_device "{scene[1]}" \
    #         --port 6030 \
    #         --checkpoint_iterations 30000 \
    #         --test_iterations 2000 7000 15000 30000 \
    #         --save_iterations 15000 30000'
    # print(cmd)
    # os.system(cmd)

    ########### 渲染 ############
    # 可选：--skip_train --skip_test
    print("--------------------------------------------------------------")
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
            python render.py \
            -m output_gt_traingulate_1600/{scene[0]} \
            --skip_test'
    print(cmd)
    os.system(cmd)

    ############ 评测 ############
    print('---------------------------------------------------------------------------------')
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
                python metrics.py \
                -m output_gt_traingulate_1600/{scene[0]}'
    print(cmd)
    os.system(cmd)