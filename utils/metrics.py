# -*- encoding: utf-8 -*-


# Evaluation metrics (Precision@1 and mAP) given similarity matrix
# Similarity matrix must have size 'test set size' x 'ref set size'
# and contains in each row the similarity of that test (query) image
# with all ref images
def precision1(sim, test_set, ref_set, kth=1):
    total = sim.size(0)
    if kth <= 1:
        max_sim, max_idx = sim.max(1)
    else:
        max_sim, max_idx = sim.kthvalue(sim.size(1) - kth + 1, 1)
    max_label = []
    for i in range(sim.size(0)):
        # get label from ref set which obtained highest score
        max_label.append(ref_set[max_idx[i, 0]][1])
    correct = sum(test_label == max_label[j] for j, (_, test_label, _) in enumerate(test_set))
    return float(correct) / total, correct, total, max_sim, max_label


# according to Oxford buildings dataset definition of AP
# the kth argument allows to ignore the k highest ranked elements of ref set
# this is used to compute AP even for the train set against train set
def avg_precision(sim, i, test_set, ref_set, kth=1):
    test_label = test_set[i][1]
    n_pos = sum(test_label == ref_label for _, ref_label, _ in ref_set)
    n_pos -= (kth - 1)
    if n_pos <= 0:
        return None
    old_recall, old_precision, ap = 0.0, 1.0, 0.0
    intersect_size, j = 0, 0
    _, ranked_list = sim[i].sort(dim=0, descending=True)
    for n, k in enumerate(ranked_list):
        if n + 1 < kth:
            continue
        if ref_set[k][1] == test_label:
            intersect_size += 1

        recall = intersect_size / float(n_pos)
        precision = intersect_size / (j + 1.0)
        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
        old_recall, old_precision = recall, precision
        j += 1
    return ap


def mean_avg_precision(sim, test_set, ref_set, kth=1):
    aps = []
    for i in range(sim.size(0)):
        # compute ap for each test image
        ap = avg_precision(sim, i, test_set, ref_set, kth)
        if ap is not None:
            aps.append(ap)
    return sum(aps) / float(len(aps))
