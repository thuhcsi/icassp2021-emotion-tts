import numpy as np


def get_metric(prs, gts, is_print=False):
    # print confusion matrix
    # return (accuracy, precision, recall, F1)
    assert len(prs) == len(gts)
    sample_num = len(prs)
    pr_np = np.array(prs)
    gt_np = np.array(gts)
    tp = int(np.sum(pr_np * gt_np))
    fp = np.sum(pr_np) - tp
    fn = np.sum(gt_np) - tp
    tn = int(np.sum((1 - pr_np) * (1 - gt_np)))

    corrected_predict = np.sum(pr_np == gt_np)
    acc = corrected_predict / sample_num
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if is_print:
        print('%s\t%s' % ('TP', 'FP'))
        print('%s\t%s' % ('FN', 'TN'))
        print('%d\t%d' % (tp, fp))
        print('%d\t%d' % (fn, tn))
        print('accuracy %.3f, precision %.3f, recall %.3f, f1 %.3f' % (
            acc, precision, recall, f1))
    return acc, precision, recall, f1


def get_uar(prs, gts):
    assert len(prs) == len(gts)
    sample_num = len(prs)
    pr_np = np.array(prs)
    gt_np = np.array(gts)
    gt_pos_num = float(np.sum(gt_np))
    gt_neg_num = sample_num - gt_pos_num
    tp = float(np.sum(pr_np * gt_np))
    tn = float(np.sum((1 - pr_np) * (1 - gt_np)))
    uar = (tp / gt_pos_num + tn / gt_neg_num) / 2
    return uar


def get_ua_wa(prs, gts, is_print=False, result_f=None):
    assert len(prs) == len(gts)
    sample_num = len(prs)
    pr_np = np.array(prs)
    gt_np = np.array(gts)
    matrix = np.zeros((2, 2))
    for pr, gt in zip(pr_np, gt_np):
        matrix[pr, gt] = matrix[pr, gt] + 1

    pos_num = float(np.sum(gt_np))
    neg_num = float(np.sum(1 - gt_np))
    wa = (matrix[0, 0] + matrix[1, 1]) / sample_num
    ua = (matrix[0, 0] / neg_num + matrix[1, 1] / pos_num) / 2
    if is_print:
        print('p\\g \t neg \t pos')
        print('neg \t %d \t %d' % (matrix[0, 0], matrix[0, 1]))
        print('pos \t %d \t %d' % (matrix[1, 0], matrix[1, 1]))
        print('ua %.3f , wa %.3f' % (ua, wa))
    if result_f:
        with open(result_f, 'a') as out_f:
            print('p\\g \t neg \t pos', file=out_f)
            print('neg \t %d \t %d' % (matrix[0, 0], matrix[0, 1]), file=out_f)
            print('pos \t %d \t %d' % (matrix[1, 0], matrix[1, 1]), file=out_f)
            print('ua %.3f , wa %.3f' % (ua, wa), file=out_f)
            print(file=out_f)
    return ua, wa
