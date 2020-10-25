import os
import math
import time
import argparse
import traceback
import numpy as np
import tensorflow as tf
from datetime import datetime


from tfr_dset import TFDataSet
from text import sequence_to_text
from utils import audio, plot, infolog, ValueWindow, debug

from sygst_hparams import hp
from models.sygst_tacotron2 import Tacotron2SYGST

log = infolog.log
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


_max_step = 500000
hdfs_ckpts='hdfs://haruna/home/byte_speech_sv/user/caixiong/ckpts'

# spec_length max = 1116
# text length max = 99


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def debug_data(batch=32, time_in=100, time_out=500):
    text_x = np.random.randint(0, 150, size=(batch, time_in), dtype=np.int32)
    mel = np.random.randn(batch, time_out, 80).astype(np.float32)
    spec = np.random.randn(batch, time_out, 1025).astype(np.float32)
    spec_len = np.random.randint(time_out // 2, time_out, size=batch, dtype=np.int32)
    aro_label = np.random.rand(batch, 2).astype(np.float32)
    val_label = np.random.rand(batch, 2).astype(np.float32)

    print('text_input:', text_x[0], 'spec_len:', spec_len, sep='\n')
    return text_x, mel, spec, spec_len, aro_label, val_label


def train(log_dir, args):
    checkpoint_path = os.path.join(hdfs_ckpts, log_dir, 'model.ckpt')
    log(hp.to_string(), is_print=False)
    log('Loading training data from: %s' % args.tfr_dir)
    log('Checkpoint path: %s' % checkpoint_path)
    log('Using model: sygst tacotron2')

    tf_dset = TFDataSet(hp, args.tfr_dir)
    feats = tf_dset.get_train_next()
    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    training = tf.placeholder_with_default(True, shape=(), name='training')
    with tf.name_scope('model'):
        model = Tacotron2SYGST(hp)
        model(feats['inputs'],
              mel_inputs=feats['mel_targets'],
              spec_inputs=feats['linear_targets'],
              spec_lengths=feats['spec_lengths'],
              ref_inputs=feats['mel_targets'],
              ref_lengths=feats['spec_lengths'],
              arousal_labels=feats['soft_arousal_labels'],
              valence_labels=feats['soft_valance_labels'],
              training=training)
        """
        text_x, mel_x, spec_x, spec_len, aro, val = debug_data(2, 5, 10)
        model(text_x, mel_x, spec_x, spec_len, mel_x, spec_len, aro, val, training=training)
        """
        model.add_loss()
        model.add_optimizer(global_step)
        stats = model.add_stats()

    # Bookkeeping:
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=2)

    # Train!
    config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            if args.restore_step:
                # Restore from a checkpoint if the user requested it.
                restore_path = '%s-%s' % (checkpoint_path, args.restore_step)
                saver.restore(sess, restore_path)
                log('Resuming from checkpoint: %s' % restore_path, slack=True)
            else:
                log('Starting a new training run ...', slack=True)

            """
            fetches = [global_step, model.optimize, model.loss, model.mel_loss, model.spec_loss,
                       model.stop_loss, model.arousal_loss, model.valence_loss, model.mel_grad_norms_max,
                       model.spec_grad_norms_max, model.stop_grad_norms_max, model.aro_grad_norms_max, model.val_grad_norms_max]
            """
            fetches = [global_step, model.optimize, model.loss, model.mel_loss, model.spec_loss,
                       model.stop_loss, model.arousal_loss, model.valence_loss]
            for _ in range(_max_step):
                start_time = time.time()
                sess.run(debug.get_ops())
                # step, _, loss, mel_loss, spec_loss, stop_loss, aro_loss, val_loss, mel_g, spec_g, stop_g, aro_g, val_g = sess.run(fetches)
                step, _, loss, mel_loss, spec_loss, stop_loss, aro_loss, val_loss = sess.run(fetches)
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                """
                message = 'Step %-7d [%.3f sec/step,ml=%.3f,spl=%.3f,sl=%.3f,al=%.3f,vl=%.3f,mg=%.4f,spg=%.4f,sg=%.4f,ag=%.4f,vg=%.4f]' % (
                    step, time_window.average, mel_loss, spec_loss, stop_loss, aro_loss, val_loss, mel_g, spec_g, stop_g, aro_g, val_g)
                """
                message = 'Step %-7d [%.3f sec/step,ml=%.3f,spl=%.3f,sl=%.3f,al=%.3f,vl=%.3f]' % (
                    step, time_window.average, mel_loss, spec_loss, stop_loss, aro_loss, val_loss)
                log(message, slack=(step % args.checkpoint_interval == 0))

                if loss > 100 or math.isnan(loss):
                    log('Loss exploded to %.5f at step %d!' % (loss, step), slack=True)
                    raise Exception('Loss Exploded')

                if step % args.summary_interval == 0:
                    log('Writing summary at step: %d' % step)
                    try:
                        summary_writer.add_summary(sess.run(stats), step)
                    except Exception as e:
                        log(f'summary failed and ignored: {str(e)}')

                if step % args.checkpoint_interval == 0:
                    log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)
                    log('Saving audio and alignment...')
                    gt_mel, gt_spec, seq, mel, spec, align = sess.run([model.mel_targets[0], model.spec_targets[0],
                                                                       model.text_targets[0], model.mel_outputs[0],
                                                                       model.spec_outputs[0], model.alignment_outputs[0]])
                    text = sequence_to_text(seq)
                    wav = audio.inv_spectrogram(hp, spec.T)
                    wav_path = os.path.join(log_dir, 'step-%d-audio.wav' % step)
                    mel_path = os.path.join(log_dir, 'step-%d-mel.png' % step)
                    spec_path = os.path.join(log_dir, 'step-%d-spec.png' % step)
                    align_path = os.path.join(log_dir, 'step-%d-align.png' % step)
                    info = '%s, %s, step=%d, loss=%.5f\n %s' % (args.model, time_string(), step, loss, text)
                    plot.plot_alignment(align, align_path, info=info)
                    plot.plot_mel(mel, mel_path, info=info, gt_mel=gt_mel)
                    plot.plot_mel(spec, spec_path, info=info, gt_mel=gt_spec)
                    audio.save_wav(hp, wav, wav_path)
                    log('Input: %s' % text)

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--log', '-l', default='')
    parser.add_argument('--restore_step', '-r', default=None)
    parser.add_argument('--tfr_dir', default='bc2013/training/tfrs_with_emo_feature')
    args = parser.parse_args()

    args.model = 'sygst_taco2'
    args.summary_interval = 200
    args.checkpoint_interval = 5000
    # args.summary_interval = 2
    # args.checkpoint_interval = 5

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_dir = 'sygst_logs' + ('_' + args.log if args.log else '')
    os.makedirs(log_dir, exist_ok=True)

    tf.set_random_seed(hp.random_seed)
    infolog.init(os.path.join(log_dir, 'train.log'), args.model)

    train(log_dir, args)


if __name__ == '__main__':
    main()
