# ViT-B/32
job_name="xclip_didemo_vit32"
DATA_PATH="[Your DiDeMo data and videos path]"
python -m torch.distributed.launch --nproc_per_node=4 \
    main_xclip.py --do_train --num_thread_reader=4 \
    --epochs=20 --batch_size=64 --n_display=10 \
    --data_path ${DATA_PATH}/DiDeMo \
    --features_path ${DATA_PATH}/DiDeMo/Video \
    --output_dir ckpts_dsw/${job_name} \
    --lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 24 \
    --datatype didemo \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}