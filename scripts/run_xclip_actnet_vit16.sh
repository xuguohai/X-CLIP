# ViT-B/16
job_name="xclip_actnet_vit16"
DATA_PATH="[Your ActivityNet data and videos path]"
python -m torch.distributed.launch --nproc_per_node=8 \
    main_xclip.py --do_train --num_thread_reader=4 \
    --epochs=20 --batch_size=32 --n_display=50 \
    --data_path ${DATA_PATH}/ActivityNet \
    --features_path ${DATA_PATH}/ActivityNet/Activity_Videos \
    --output_dir ckpts_dsw/${job_name} \
    --lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 \
    --datatype activity \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a log/${job_name}