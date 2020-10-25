import six
import json


# hyper parameter util class
class HParams:
    def __init__(self, **kwargs):
        """A simple alternative implementation for tf.contrib.training.HParams

        # Arguments
            kwargs: all key word parameters which will be added as instance atrributes
                used as hyper parameters
        """
        for k, v in six.iteritems(kwargs):
            self.add_hparam(k, v)

    def add_hparam(self, name, value):
        """add a new hyperparameter given a name and value

        if name is an existed hyperparameter, then it's value is
            updated as the new value

        # Arguments
            name: str name of the new hyperparameter will be added
            value: the value of new hyperparameter
        """
        setattr(self, name, value)

    def del_hparam(self, name):
        """delete a hyperparameter named name

        # Arguments
            name: str name of the hyperparameter will be deleted
        """
        delattr(self, name)

    def update(self, D, **kwargs):
        """update or add hyper parameters

        # Arguments
            D: a object that has keys() method, or can be iterated
                as for k, v in D
            kwargs: extra key workd arguments for update hyper patameters
        """
        self.__dict__.update(D, **kwargs)

    def parse(self, values):
        """parse a str that is splited by ';' and update them into attributes

        Note: we use ';' as the delimiter not ',' as in tf.contrib.training.HParams
            because ',' will conflict the delimiter ',' in  list and dict

        # Arguments
            values: a str contains hyper parameters which is splited with ';'
                and paired with '=', e.g., 'epochs=20,learning_rate=0.001'
        """
        pairs = values.split(";")
        pairs = [x.strip().split("=") for x in pairs if x.strip() and '=' in x]
        dict_pairs = dict(pairs)
        for k in dict_pairs:
            if k not in self.__dict__:
                raise KeyError('can not parse a not existing hyperparameter:"{}"'.format(k))
            # self.__dict__[k] = type(self.__dict__[k])(dict_pairs[k])    # 还无法解析字典和列表元素
            try:
                v = json.loads(dict_pairs[k])    # note: 参数值如果是字典, 则该字典的key只能是字符串(json要求)
            except json.JSONDecodeError:
                v = json.loads('"' + dict_pairs[k] + '"')  # 直接解析字符串hello会报错, 必须解析"hello"才可以
            self.__dict__[k] = v
        return self

    def print(self):
        """this func prints all hyper parameters"""
        print('\n\n')
        print('--------------------------------------------------')
        print('All Hyper Parameters:')
        print('--------------------------------------------------')
        hps = self.__dict__
        for hp in hps:
            print('  {}={}'.format(hp, hps[hp]))
        print('--------------------------------------------------')
        print('\n\n')

    def to_string(self):
        hp = '\n'
        hp += '--------------------------------------------------\n'
        hp += 'All Hyper Parameters:\n'
        hp += '--------------------------------------------------\n'
        hps = self.__dict__
        for k in hps:
            hp += '  {}={}\n'.format(k, hps[k])
        hp += '--------------------------------------------------\n'
        hp += '\n'
        return hp
