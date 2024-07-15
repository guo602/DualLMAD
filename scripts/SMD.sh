feature_dim=38
seq_len=100

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /data/home/nealyguo/Datasets/SMD_ONE/SMD \
  --model_id dualLMAD \
  --model dualLMAD \
  --data SMD \
  --features M \
  --feature_dim $feature_dim \
  --seq_len $seq_len \
  --pred_len 0 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 8 \
  --enc_in $feature_dim \
  --c_out_im $seq_len \
  --c_out_time $feature_dim \
  --anomaly_ratio 2 \
  --batch_size 64 \
  --learning_rate 0.01 \
  --train_epochs 5 \
  --step 100 \
  --cos 1 \
  --is_pretrain 1\
  --devices 0