# resotre_ckpt
# feat_npy_fold
# feat_npy_name
# eval_tfr
# config_file

## mmd

python mmd_eval_feats.py \
 --gpu='7' \
 --config_file=./mmd_cfg/iemocap2recola.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/iemocap/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd_alpha0.5/iemocap2recola/mmd_valance_02012127_best_params_ckpt/model-2670 \
 --feat_npy_fold=./npys/mmd/iemocap2recola/valance \
 --feat_npy_name=iemocap.npy

python mmd_eval_feats.py \
 --gpu='7' \
 --config_file=./mmd_cfg/iemocap2recola.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/recola/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd_alpha0.5/iemocap2recola/mmd_valance_02012127_best_params_ckpt/model-2670 \
 --feat_npy_fold=./npys/mmd/iemocap2recola/valance \
 --feat_npy_name=recola.npy

python mmd_eval_feats.py \
 --gpu='7' \
 --config_file=./mmd_cfg/recola2iemocap.yml \
 --label_type=arousal \
 --eval_tfr=/home/ddy17/ser_data/cx_data/iemocap/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd_alpha0.5/recola2iemocap/mmd_arousal_02011733_best_params_ckpt/model-420 \
 --feat_npy_fold=./npys/mmd/recola2iemocap/arousal \
 --feat_npy_name=iemocap.npy

python mmd_eval_feats.py \
 --gpu='7' \
 --config_file=./mmd_cfg/recola2iemocap.yml \
 --label_type=arousal \
 --eval_tfr=/home/ddy17/ser_data/cx_data/recola/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd_alpha0.5/recola2iemocap/mmd_arousal_02011733_best_params_ckpt/model-420 \
 --feat_npy_fold=./npys/mmd/recola2iemocap/arousal \
 --feat_npy_name=recola.npy

python mmd_eval_feats.py \
 --gpu='7' \
 --config_file=./mmd_cfg/recola2iemocap.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/iemocap/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd_alpha0.5/recola2iemocap/mmd_valance_02011857_best_params_ckpt/model-690 \
 --feat_npy_fold=./npys/mmd/recola2iemocap/valance \
 --feat_npy_name=iemocap.npy

python mmd_eval_feats.py \
 --gpu='7' \
 --config_file=./mmd_cfg/recola2iemocap.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/recola/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd_alpha0.5/recola2iemocap/mmd_valance_02011857_best_params_ckpt/model-690 \
 --feat_npy_fold=./npys/mmd/recola2iemocap/valance \
 --feat_npy_name=recola.npy