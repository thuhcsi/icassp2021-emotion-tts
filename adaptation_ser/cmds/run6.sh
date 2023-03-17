#
## GST2MMD alpha 0.5
## iemo2reco arousal
#python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/iemocap2recola/gst2mmd_arousal_02011602_best_params_ckpt/model-2550 --result_file='gst2mmd05_iemo2reco_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/iemocap2recola/gst2mmd_arousal_02011639_best_params_ckpt/model-2550 --result_file='gst2mmd05_iemo2reco_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/iemocap2recola/gst2mmd_arousal_02020111_best_params_ckpt/model-2460 --result_file='gst2mmd05_iemo2reco_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/iemocap2recola/gst2mmd_arousal_02020149_best_params_ckpt/model-2850 --result_file='gst2mmd05_iemo2reco_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/iemocap2recola/gst2mmd_arousal_02020227_best_params_ckpt/model-2850 --result_file='gst2mmd05_iemo2reco_arousal.txt'
#
## reco2iemo arousal
#python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/recola2iemocap/gst2mmd_arousal_02011602_best_params_ckpt/model-720 --result_file='gst2mmd05_reco2iemo_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/recola2iemocap/gst2mmd_arousal_02011610_best_params_ckpt/model-780 --result_file='gst2mmd05_reco2iemo_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/recola2iemocap/gst2mmd_arousal_02011619_best_params_ckpt/model-150 --result_file='gst2mmd05_reco2iemo_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/recola2iemocap/gst2mmd_arousal_02011627_best_params_ckpt/model-690 --result_file='gst2mmd05_reco2iemo_arousal.txt'
#
#python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='gst2mmd' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/gst2mmd_alpha0.5/recola2iemocap/gst2mmd_arousal_02011636_best_params_ckpt/model-810 --result_file='gst2mmd05_reco2iemo_arousal.txt'

#python mmd_train.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola
#python mmd_train.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola
#python mmd_train.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola
#python mmd_train.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola
#python mmd_train.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola
#
#python mmd_train.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap
#python mmd_train.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap
#python mmd_train.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap
#python mmd_train.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap
#python mmd_train.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --label_type='arousal' --model_type='mmd2' --ckpt_base_dir=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap

# recola2iemocap, arousal
python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap/mmd2_arousal_02231855_best_params_ckpt/model-450 --result_file='r2_mmd05_recola2iemocap_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap/mmd2_arousal_02231904_best_params_ckpt/model-450 --result_file='r2_mmd05_recola2iemocap_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap/mmd2_arousal_02231912_best_params_ckpt/model-690 --result_file='r2_mmd05_recola2iemocap_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap/mmd2_arousal_02231921_best_params_ckpt/model-570 --result_file='r2_mmd05_recola2iemocap_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/recola2iemocap.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/recola2iemocap/mmd2_arousal_02231929_best_params_ckpt/model-360 --result_file='r2_mmd05_recola2iemocap_arousal.txt'

# iemocap2recola, arousal
python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola/mmd2_arousal_02231547_best_params_ckpt/model-2400 --result_file='r2_mmd05_iemocap2recola_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola/mmd2_arousal_02231626_best_params_ckpt/model-1770 --result_file='r2_mmd05_iemocap2recola_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola/mmd2_arousal_02231705_best_params_ckpt/model-3990 --result_file='r2_mmd05_iemocap2recola_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola/mmd2_arousal_02231742_best_params_ckpt/model-1980 --result_file='r2_mmd05_iemocap2recola_arousal.txt'

python mmd_eval.py --config_file=./mmd_cfg/iemocap2recola.yml --gpu='6' --model_type='mmd2' --label_type='arousal' --restore_ckpt=/home/ddy17/experiments/gst_ser/mmd2_alpha0.5/iemocap2recola/mmd2_arousal_02231819_best_params_ckpt/model-3300 --result_file='r2_mmd05_iemocap2recola_arousal.txt'

