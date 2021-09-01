import os
import re
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

from text import text_to_sequence
from synthesizer import Synthesizer

_pad_mel = 0.
_pad_text = 0
_seed = 2020

sygst_atten_files = {
    'a0': 'sygst_emo_data/emo2d_mel_gst_weights/arousal0.npy',
    'a1': 'sygst_emo_data/emo2d_mel_gst_weights/arousal1.npy',
    'v0': 'sygst_emo_data/emo2d_mel_gst_weights/valence0.npy',
    'v1': 'sygst_emo_data/emo2d_mel_gst_weights/valence1.npy',
}

emb_atten_files = {
    'a0': 'embgst_emo_data/emo2d_mel_gst_weights/arousal0.npy',
    'a1': 'embgst_emo_data/emo2d_mel_gst_weights/arousal1.npy',
    'v0': 'embgst_emo_data/emo2d_mel_gst_weights/valence0.npy',
    'v1': 'embgst_emo_data/emo2d_mel_gst_weights/valence1.npy',
}

emo_atten_files = {
    'emo0': 'emogst_emo_data/emo_gst_weights/emo0.npy',
    'emo1': 'emogst_emo_data/emo_gst_weights/emo1.npy',
    'emo2': 'emogst_emo_data/emo_gst_weights/emo2.npy',
    'emo3': 'emogst_emo_data/emo_gst_weights/emo3.npy',
}


def read_meta_from_file(text_file):
    with open(text_file, 'r') as fr:
        lines = fr.readlines()
    lines = [line.strip() for line in lines if line.strip()]  # remove blank lines
    lines = [line for line in lines if line[0] != '#']        # remove comment lines

    try:
        texts, idx = [], 1
        while lines[idx] != '--emo_strs':
            texts.append(lines[idx])
            idx += 1

        emo_strs, idx = [], idx + 1
        while lines[idx] != '--ref_mels':
            emo_strs.append(lines[idx])
            idx += 1

        ref_mels, idx = [], idx + 1
        while lines[idx] != '--gta_mels':
            ref_mels.append(lines[idx])
            idx += 1

        gta_mels, idx = [], idx + 1
        while lines[idx] != '--end':
            gta_mels.append(lines[idx])
            idx += 1
    except IndexError as e:
        print('The format of text file is not correct')
        raise e
    return texts, emo_strs, ref_mels, gta_mels


def process_texts(args, texts, cleaners=['english_cleaners']):
    '''Convert texts to char id seqs, then batch and pad them
    # Returns
        A list of batch padded text seqs, each batch is a int32 np.array
    '''
    def text_to_name(text):
        text = re.sub(r'\W+', '_', text.strip()) 
        text = re.sub(r'_+', '_', text)
        text = re.sub(r'_$', '', text)
        name = f'eval-{args.eval_step}' + '-emo-{}_' + text
        return os.path.join(args.output_dir, name)

    batch_size, num_texts = args.batch_size, len(texts)
    batched_seqs, batched_texts, batched_names = [], [], []

    names = [text_to_name(text) for text in texts]
    seqs = [text_to_sequence(text, cleaners) for text in texts]

    for i in range(0, num_texts, batch_size):
        end = min(i + batch_size, num_texts)
        batch_seqs = seqs[i: end]
        batch_texts = texts[i: end]
        batch_names = names[i: end]
        batch_maxlen = len(max(batch_seqs, key=len))

        def pad_fn(seq):
            return seq + [_pad_text] * (batch_maxlen - len(seq))
        batch_seqs = list(map(pad_fn, batch_seqs))
        batch_seqs = np.array(batch_seqs, dtype=np.int32)

        batched_seqs.append(batch_seqs)
        batched_texts.append(batch_texts)
        batched_names.append(batch_names)
    return batched_seqs, batched_texts, batched_names


def process_emostrs(args, emo_strs):
    '''Get attention weights from emo strs
    # Returns
        A list of attention weights arrays, each array with dtype np.float32
            and shape [batch_size, num_heads, 1, num_tokens] for sygst model.
            And for embgst model, each list item is a 2-tuple of weights arrays
            with same shape described above for arousal and valence separately.
    '''
    if args.model_name == 'taco2':
        return None

    def load_av_weights(emo_str):
        atten_files = sygst_atten_files if model_name == 'sygst' else emb_atten_files
        emos = [float(emo) for emo in emo_str.split('-')]
        a_prob = np.clip(emos[0], 0., 1.)
        v_prob = np.clip(emos[1], 0., 1.)
        a0_weights = np.load(atten_files['a0'])
        a1_weights = np.load(atten_files['a1'])
        v0_weights = np.load(atten_files['v0'])
        v1_weights = np.load(atten_files['v1'])

        aro_weights = (1 - a_prob) * a0_weights + a_prob * a1_weights
        val_weights = (1 - v_prob) * v0_weights + v_prob * v1_weights
        if args.model_name == 'sygst':
            aro_weights, aro_other = np.split(aro_weights, 2, axis=0)
            val_other, val_weights = np.split(val_weights, 2, axis=0)
            atten_weights = np.concatenate((aro_weights, val_weights), axis=0)
            atten_weights = np.tile(atten_weights, [batch_size, 1, 1, 1])
            return atten_weights
        else:
            aro_weights = np.tile(aro_weights, [batch_size, 1, 1, 1])
            val_weights = np.tile(val_weights, [batch_size, 1, 1, 1])
            return (aro_weights, val_weights)

    def load_emo_weights(emo_str):
        emo_type, emo_strength = emo_str.split('-') if '-' in emo_str else (emo_str, None)
        assert emo_type in ['0', '1', '2', '3']
        atten_weights = np.load(emo_atten_files['emo' + emo_type])
        if emo_strength is not None:
            try:
                emo_strength = float(emo_strength)
            except Exception:
                raise ValueError('emo_strength must be None or a float number')
            atten_weights *= emo_strength
        return np.tile(atten_weights, [batch_size, 1, 1, 1])

    model_name, batch_size = args.model_name, args.batch_size
    load_fun = load_emo_weights if model_name == 'emogst' else load_av_weights
    return list(map(load_fun, emo_strs))


def process_mels(mel_names, batch_size=None):
    '''Get mels from mel names
    # Arguments
        mel_names: A list of mel file names for loading mels
        batch_size: The batch size, default(None) is 1 for refrences mels and
            >1 for gta mels. Since reference mels is used likewisely emo strs
            each reference mel used for synthesizing all texts.

    # Returns
        A list of zipped batch-padded mel arrays and mel lengths
    '''
    mels = list(map(np.load, mel_names))
    batched_mels, batched_lengths, num_mels = [], [], len(mels)
    batch_size = batch_size if batch_size else 1
    for i in range(0, num_mels, batch_size):
        end = min(i + batch_size, num_mels)
        batch_mels = mels[i: end]
        batch_maxlen = len(max(batch_mels, key=len))  # mel [time_step, 80]
        batched_lengths.append(np.array(list(map(len, batch_mels)), dtype=np.int32))

        def pad_fn(mel):
            return np.pad(mel, [[0, batch_maxlen - len(mel)], [0, 0]],
                          mode='constant', constant_values=_pad_mel)
        batch_mels = list(map(pad_fn, batch_mels))
        batch_mels = np.array(batch_mels, dtype=np.float32)
        batched_mels.append(batch_mels)
    zipped_mels = zip(batched_mels, batched_lengths)
    return zipped_mels


def prepare_run(args):
    if os.path.isfile(args.texts):
        texts, emo_strs, ref_names, gta_names = read_meta_from_file(args.texts)
    else:
        assert args.texts is not None
        texts = [text.strip() for text in args.texts.split('|')]
        emo_strs = [emo.strip() for emo in args.emo_strs.split('|')] if args.emo_strs else []
        ref_names = [ref.strip() for ref in args.ref_mels.split('|')] if args.ref_mels else []
        gta_names = [gta.strip() for gta in args.gta_mels.split('|')] if args.gta_mels else []
    args.texts, args.emo_strs, args.ref_names, args.gta_names = texts, emo_strs, ref_names, gta_names

    assert not (ref_names and gta_names)
    assert not gta_names or len(gta_names) == len(texts)
    if args.model_name == 'taco2':
        assert not (emo_strs or ref_names)
    else:
        assert emo_strs or ref_names or gta_names

    tf.set_random_seed(_seed)
    args.eval_step = int(re.search(r'ckpt-(\d+)', args.ckpt_path).group(1))
    args.output_dir = os.path.join(args.output_dir,
                                   datetime.now().strftime('%Y%m%d_%H%M'))
    os.makedirs(args.output_dir, exist_ok=True)

    args.batch_seqs, args.batch_texts, args.batch_names = process_texts(args, texts)
    args.emo_weights = None if not emo_strs else process_emostrs(args, emo_strs)
    args.ref_inputs = None if not ref_names else process_mels(ref_names, None)
    args.gta_inputs = None if not gta_names else process_mels(gta_names, args.batch_size)
    args.use_att = (args.emo_weights is not None) and not ref_names and (not gta_names or args.gta_att)
    args.use_ref = args.ref_inputs is not None
    args.use_gta = args.gta_inputs is not None


def run_eval(args):

    synth = Synthesizer(args.use_gta, args.use_ref, args.use_att, args.model_name)
    synth.load(args.ckpt_path)

    batch_size = args.batch_size
    num_batches = len(args.batch_texts)
    use_att, use_ref, use_gta = args.use_att, args.use_ref, args.use_gta

    gta_mels_lens = args.gta_inputs if args.use_gta else [None, None]
    emos = args.ref_inputs if args.use_ref else args.emo_weights  # 优先使用ref mel
    emos_info = args.ref_names if args.use_ref else args.emo_strs
    emos, emos_info = [emos, emos_info] if emos else [[None], [None]]

    def print_infos():
        print(f'\nLoading checkpoint: {args.ckpt_path}')
        print('\nSynthesis Infos:\n  ', end='')
        print(f'use_att={use_att}', f'use_ref={use_ref}', f'use_gta={use_gta}',
              f'num_batches={num_batches}', f'batch_size={batch_size}',
              f'num_texts={len(args.texts)}', f'emo_infos={emos_info}',
              f'model={args.model_name}', f'output_dir={args.output_dir}',
              sep='\n  ')
    print_infos()

    for emo, emo_info in zip(emos, emos_info):
        print(f'\nSynthesizing with emo info: {emo_info} ...')
        for i in range(num_batches):
            batch_seq = args.batch_seqs[i]
            batch_text = args.batch_texts[i]
            if os.path.isfile(emo_info):
                emo_info = os.path.splitext(os.path.basename(emo_info))[0]   # when mel as ref audio, only preserve the mel file name
            batch_name = [n.format(emo_info) for n in args.batch_names[i]]
            print(f'  Synthesizing {i + 1}th batch with sentences:', end='')
            print('', *batch_text, sep='\n    ')
            call_fn_kwargs = {'mel_inputs': gta_mels_lens[0], 'mel_lengths': gta_mels_lens[1]}
            if emo is not None:
                emo = emo if isinstance(emo, (tuple, list)) else [emo]
                emo = [x[: len(batch_text)] for x in emo]
            if args.use_ref:
                call_fn_kwargs.update(ref_inputs=emo[0], ref_lengths=emo[1])
            elif args.use_att:
                if args.model_name in ['sygst', 'emogst']:
                    call_fn_kwargs.update(atten_weights=emo[0])
                elif args.model_name == 'embgst':
                    call_fn_kwargs.update(aro_weights=emo[0], val_weights=emo[1])
            synth.synthesize(batch_seq, batch_text, batch_name, **call_fn_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', default=None, help='Model name[taco2 sygst embgst]')
    parser.add_argument('--ckpt_path', '-c', default=None, help='Path to model checkpoint')
    parser.add_argument('--texts', '-t', default=None, help='Test text sentences sperated by "|"')
    parser.add_argument('--ref_mels', '-r', default=None, help='Reference mels for gst synthesis')
    parser.add_argument('--emo_strs', '-e', default='1.0-0.5', help='Format aro_ratio-val_ratio')
    parser.add_argument('--gta_mels', '-g', default=None, help='Ground truth alignment synthesis')
    parser.add_argument('--gta_att', '-a', action='store_true', help='Gta mode use emo weights')
    parser.add_argument('--batch_size', '-b', default=1, type=int, help='Batch size for synthesis')
    parser.add_argument('--output_dir', '-o', default=None, help='Path for saving outputs')
    args = parser.parse_args()

    assert args.model_name in ['taco2', 'sygst', 'emogst', 'embgst']

    if not args.ckpt_path:
        args.ckpt_path = './sygst_logs_02/model.ckpt-165000'
        # args.ckpt_path = './sygst_logs/model.ckpt-65000'
        # args.ckpt_path = './sygst_logs_feed_all_frame/model.ckpt-50000'
        # args.ckpt_path = '/home/caixiong/l3/model.ckpt-235000'
        # args.ckpt_path = '/home/caixiong/l1/model.ckpt-120000'
        # args.ckpt_path = 'taco2_logs_impute_false_mae_mask_Fasle_none/model.ckpt-210000'
        # args.ckpt_path = 'taco2_logs_impute_false_mae_mask_false_relu/model.ckpt-180000'
        # args.ckpt_path = 'sygst_emo_data/emo2d_ckpts/model.ckpt-250000'
    if args.ckpt_path.isdigit():
        args.ckpt_path = f'{args.model_name}_emo_data/ckpts/model.ckpt-{args.ckpt_path}'
    if not args.texts:
        # args.texts = "Yesterday's records can keep us from repeating yesterday's mistakes."
        args.emo_strs = '0|1|2|3' if args.model_name == 'emogst' else '1.0-0.5|0.0-0.5|0.0-1.0|0.0-0.0'
        # args.emo_strs = '0-1.2|1-1.2|2-1.2|3-1.2|0-0.8|1-0.8|2-0.8|3-0.8'
        args.texts = """And when they had set them in the midst they asked By what power or by what name have ye done this?|
                        he said not to be as like to old friends as new ones.|
                        he came out from under the trees to other hills."""
        args.texts = "Too young to simple, sometimes naive.|He thought it was time to present the present.|I visited museums and sat in public gardens."
    if not args.output_dir:
        args.output_dir = args.model_name + '_outputs'

    prepare_run(args)
    run_eval(args)


if __name__ == '__main__':
    main()
