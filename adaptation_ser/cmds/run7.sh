python gst_eval_feats.py \
 --gpu='7' \
 --config_file=./gst_cfg/iemocap2recola.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/iemocap/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/cr_baseline/iemocap2recola/cr_valance_01301758_best_params_ckpt/model-1860 \
 --feat_npy_fold=./npys/baseline/iemocap2recola/valance \
 --feat_npy_name=iemocap.npy

python gst_eval_feats.py \
 --gpu='7' \
 --config_file=./gst_cfg/iemocap2recola.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/recola/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/cr_baseline/iemocap2recola/cr_valance_01301758_best_params_ckpt/model-1860 \
 --feat_npy_fold=./npys/baseline/iemocap2recola/valance \
 --feat_npy_name=recola.npy


 python gst_eval_feats.py \
 --gpu='7' \
 --config_file=./gst_cfg/recola2iemocap.yml \
 --label_type=arousal \
 --eval_tfr=/home/ddy17/ser_data/cx_data/iemocap/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/cr_baseline/recola2iemocap/cr_arousal_01301651_best_params_ckpt/model-540 \
 --feat_npy_fold=./npys/baseline/recola2iemocap/arousal \
 --feat_npy_name=iemocap.npy

python gst_eval_feats.py \
 --gpu='7' \
 --config_file=./gst_cfg/recola2iemocap.yml \
 --label_type=arousal \
 --eval_tfr=/home/ddy17/ser_data/cx_data/recola/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/cr_baseline/recola2iemocap/cr_arousal_01301651_best_params_ckpt/model-540 \
 --feat_npy_fold=./npys/baseline/recola2iemocap/arousal \
 --feat_npy_name=recola.npy

python gst_eval_feats.py \
 --gpu='7' \
 --config_file=./gst_cfg/recola2iemocap.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/iemocap/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/cr_baseline/recola2iemocap/cr_valance_01301643_best_params_ckpt/model-270 \
 --feat_npy_fold=./npys/baseline/recola2iemocap/valance \
 --feat_npy_name=iemocap.npy

python gst_eval_feats.py \
 --gpu='7' \
 --config_file=./gst_cfg/recola2iemocap.yml \
 --label_type=valance \
 --eval_tfr=/home/ddy17/ser_data/cx_data/recola/tfrs/all.tfr \
 --restore_ckpt=/home/ddy17/experiments/gst_ser/cr_baseline/recola2iemocap/cr_valance_01301643_best_params_ckpt/model-270 \
 --feat_npy_fold=./npys/baseline/recola2iemocap/valance \
 --feat_npy_name=recola.npy