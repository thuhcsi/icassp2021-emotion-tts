import argparse


class MyArgumentParser(argparse.ArgumentParser):

    def parse_to_dict(self, args=None, namespace=None):
        result = dict()
        args, argv = self.parse_known_args(args, namespace)
        # flags = vars(flags)
        if args is not None:
            result.update(vars(args))
        result.update(self.parse_argv(argv))
        return args, result

    @staticmethod
    def parse_argv(argv):
        argv_dict = dict()
        if argv is None:
            return argv_dict
        for ele in argv:
            ele = ele.strip('-')
            eles = ele.split('=')
            k = eles[0]
            v = eles[-1]
            if v.lower() == 'true':
                v = True

            elif v.lower() == 'false':
                v = False
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            argv_dict[k] = v
        return argv_dict
