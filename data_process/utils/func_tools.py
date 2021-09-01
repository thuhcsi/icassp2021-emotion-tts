import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split


sklearn_split = partial(train_test_split, random_state=2021)


def split_data(lines, splits, split_key=None):
    """
    # Arguments
        lines: all the meta lines needed to be split
        splits: the split ratios
        split_key: a function provided by the caller takes a line as input and
            return the sample class. It is used to split even for each class.
    # Returns
        A list has the same length as splits, where each element is a list of
            lines for this split ratio.
    """
    assert all([s > 0 for s in splits])

    def _split_normally(lines, split_ratios):     # Normally random splits
        accum, split_lines = 0., []
        for i, r in enumerate(split_ratios):
            if i == len(split_ratios) - 1:  # The last split
                split_lines.append(lines)
                break
            cur_r = r / (1 - accum)
            cur_part, lines = sklearn_split(lines, train_size=cur_r)
            accum += r
            split_lines.append(cur_part)
        return split_lines

    splits = [s / sum(splits) for s in splits]  # make sum to 1
    mapped_lines, res_lines = {}, [[] for s in splits]

    if split_key is None:
        return _split_normally(lines, splits)

    # Collect lines for each class
    for line in lines:
        cls = split_key(line)   # Sample class label
        if cls not in mapped_lines:
            mapped_lines[cls] = [line]
        else:
            mapped_lines[cls].append(line)

    # Split lines for each class and gather them to the result lines
    for cls in mapped_lines:
        # Normally split for each class
        cls_lines = _split_normally(mapped_lines[cls], splits)
        # Add split lines of current class to result lines
        for c_line, r_line in zip(cls_lines, res_lines):
            r_line.extend(c_line)

    # Final shuffle to break the fixed calss order in each split
    for s in res_lines:
        np.random.shuffle(s)

    return res_lines
