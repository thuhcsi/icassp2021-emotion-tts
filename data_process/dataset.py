import os
from os import path
from tqdm import tqdm
# from concurrent.futures import wait
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


class Dataset(object):
    """
    The base class of dataset, which provides the some standard interfaces for
    processing dataset and some common operations, such as create dirs and some
    legality checks ...
    """

    def __init__(self, hp, name, base_dir='data', data_dirs='data', **kwargs):
        """
        # Arguments
            hp: the hyper parameter object
            name: the name of dataset, preferably all lower letters. it will be
                used as the dir name of this dataset.
            base_dir: this dir include 'raw', 'processed', 'training' sub dirs.
            data_dirs: it is the dir where the copyed raw data will be saved.
                it is prefixed by self.processed_dir/datas. For speech data, it
                is usually named 'wavs', i.e., self.processed_dir/datas/wavs.
                it can be a str or a list of strs, if you have more than one
                kind of data, such as wavs and images
            kwargs: Currently, there are two key args are used, 'thread' and
                'process' respectively, which are the max number workers of
                ThreadPoolExecutor and ProcessPoolExecutor for parallel exe.

        # Exceptions
            ValueError: if the base_dir is not existed

        """
        if not path.isdir(base_dir):
            raise ValueError('The base_dir {} not existed'.format(base_dir))
        assert type(data_dirs) in [str, list, tuple]

        self.hp = hp
        self.name = name
        self.base_dir = base_dir
        self._config = {k: kwargs.pop(k, None) for k in ['thread', 'process']}
        # _config used for get max_workers for multi-threads or multi-processes

        # dirs at self.base_dir folder
        self.raw_dir = path.join(base_dir, 'raw', name)
        self.processed_dir = path.join(base_dir, 'processed', name)
        self.training_dir = path.join(base_dir, 'training', name)

        # dirs at self.processed_dir folder
        self.data_dir = path.join(self.processed_dir, 'datas')
        self.feature_dir = path.join(self.processed_dir, 'features')
        self.meta_file = path.join(self.processed_dir, 'meta.txt')
        self.log_file = path.join(self.processed_dir, 'log.txt')
        # Used for saving the hyper parameters for feature extraction

        # dirs at self.training_dir folder
        self.tf_dir = path.join(self.training_dir, 'tf_datas')

        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.tf_dir, exist_ok=True)

        self.data_dirs = []
        data_dirs = [data_dirs] if type(data_dirs) == str else data_dirs
        for name in data_dirs:
            full_dir = path.join(self.data_dir, name)
            os.makedirs(full_dir, exist_ok=True)
            self.data_dirs.append(full_dir)

    def process_raw(self):
        """
        Process the raw folder of the dataset: copy or resample the wav files
        to the 'wavs' folder and create a 'meta.txt'(with labels and sample
        numbers)

        It assumes the input dir of raw dataset at self.base_dir/raw/self.name
        and the output dir at self.base_dir/processed/self.name

        # Exceptions
            RuntimeError: if self.raw_dir is not a existed dir
        """

        if not path.isdir(self.raw_dir):
            raise RuntimeError('The raw_dir {} not existed'.format(self.raw_dir))

    def process_feature(self):
        """
        Extract various features using the 'meta.txt' and the 'data'(or other
        customized names, e.g., 'wavs') folder.

        The extracted feature is saved at the same dir with 'meta.txt', i.e,
        at self.processed_dir

        # Exceptions
            RuntimeError: if the meta.txt is not a existed.
        """
        if not path.isfile(self.meta_file):
            raise RuntimeError('The {} not existed'.format(self.meta_file))

    def process_training(self):
        pass

    def log(self, action=None, from_paths=None, to_paths=None,
            comment=None, full_print=True, compute_md5=True):
        log_str = ('\n\n\n'
                   '==================================================\n'
                   '  Index : {:04d}\n'
                   "  Action: '{}'\n"
                   '  From  : {}  To  {}\n'
                   '  Coment: {}\n'
                   '  md5sum: {}'
                   '{}\n'
                   '{:04d}\n'
                   '==================================================\n')

        def _process_md5sum(dirs, idxs=[0, -1]):
            md5sum_strs = ['\n']
            for d in dirs:
                assert path.isfile(d) or path.isdir(d)
                if path.isfile(d):
                    md5sum_strs.append(os.popen(f'md5sum {d}').read())
                    continue
                for i in idxs:
                    try:
                        fs = [path.join(d, f) for f in os.listdir(d)]
                        md5sum_strs.append(os.popen(f'md5sum {fs[i]}').read())
                    except IndexError:
                        continue
            return (' ' * 10).join(md5sum_strs)

        if from_paths is not None:
            from_paths = [from_paths] if type(from_paths) == str else from_paths
        if to_paths is not None:
            to_paths = [to_paths] if type(to_paths) == str else to_paths
        if compute_md5:
            md5sums = _process_md5sum(to_paths)
        if not full_print:
            from_paths = ', '.join([path.basename(p) for p in from_paths])
            to_paths = ', '.join([path.basename(p) for p in to_paths])

        log_exist = path.isfile(self.log_file) and path.getsize(self.log_file)
        with open(self.log_file, 'a+') as ff:
            ff.seek(0, 0)
            num = int(ff.readlines()[-2]) + 1 if log_exist else 1
            ff.write(log_str.format(num, action, from_paths, to_paths, comment,
                                    md5sums, self.hp.to_string().strip(), num))

        # done

    @staticmethod
    def para_executor(func, *args, mode='thread', num=None, desc='Processing'):
        config = dict(thread=ThreadPoolExecutor, process=ProcessPoolExecutor)
        assert mode in config
        with config[mode](max_workers=num) as p:
            res = list(tqdm(p.map(func, *args), desc=desc, total=len(args[0])))
        return res

    @staticmethod
    def write_meta(meta_file, lines):
        with open(meta_file, 'w', encoding='utf-8') as fw:
            fw.writelines(lines)
        # done
