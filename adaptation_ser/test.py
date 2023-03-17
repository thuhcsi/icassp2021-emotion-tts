

if __name__ == '__main__':
    with open('tmp2.txt', 'a') as f:
        print('hello', file=f)
    # yparams = cfg_process.YParams('./gst_cfg/prepare_data.yml', 'default')
    # parser = parser_util.MyArgumentParser()
    # argc, flags_dict = parser.parse_to_dict()
    # hpsPrecessor = cfg_process.HpsGSTPreprocessor(yparams, flags_dict)
    # hpsPrecessor.preprocess()
    # train_metas, test_metas = prepare_data.load_data(yparams)
    # mu1, std1 = prepare_data.get_mean_std(train_metas)
    # # mu2, std2 = prepare_data.get_mean_std2(train_metas)
    # print(mu1)
    # # print(mu2)
    # print(std1)
    # # print(std2)
    # train_metas, test_metas = prepare_data.norm(train_metas, test_metas)
    # mu2, std2 = prepare_data.get_mean_std(train_metas)
    # print(mu2)
    # print(std2)
