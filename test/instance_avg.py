# -*- encoding: utf-8 -*-

import torch
from utils import get_lab_indicators


def instance_avg(device, embeddings, dataset, labels, k=-1):
    # create new embeddings for the dataset, where each embedding
    # is replaced with a weighted sum of the k nearest neighbors
    # within its instance. if k is negative, all neighbors of the instance
    # are used
    sim = torch.mm(embeddings, embeddings.t())
    # for each embedding, set the similarities to embeddings of different
    # labels to -2, plus to itself, so the maximal similarities are always
    # neighbors of the same instance
    lab_ind = get_lab_indicators(dataset, device)
    new_embeddings = embeddings.clone()
    for i, (_, lab, _) in enumerate(dataset):
        num_neighbors = lab_ind[lab].sum() - 1
        if k >= 0 and k < num_neighbors:
            num_neighbors = k
        if num_neighbors <= 0:
            new_embeddings[i] = embeddings[i]
            continue
        sim[i, i] = -2
        sim[i][1 - lab_ind[lab]] = -2
        _, best_neighbors = torch.sort(sim[i], dim=0, descending=True)
        agg_embedding = embeddings[i].clone()
        for j in range(num_neighbors):
            weight = (num_neighbors - j) / float(num_neighbors + 1)
            agg_embedding += embeddings[best_neighbors[j]] * weight
        new_embeddings[i] = agg_embedding / (agg_embedding.norm() + 1e-10)
    return new_embeddings, dataset


# a method of simply averaging the descriptors for each instance
# this is less useful as it may pull outliers into the average
# def instance_avg(device, embeddings, dataset, labels):
#     # create a fictional dataset with one entry per label, with
#     # its embeddings as the average of all descriptors of each label
#     fictional_set = [(None, lab, None) for lab in labels]
#     new_embeddings = tensor(device, len(labels), embeddings.size(1))
#     avg = {lab: tensor(device, embeddings.size(1)).fill_(0)
#            for lab in labels}
#     for embedding, (_, lab, _) in zip(embeddings, dataset):
#         avg[lab] += embedding  # no need to average since we normalize
#     for i, lab in enumerate(labels):
#         new_embeddings[i] = avg[lab] / (avg[lab].norm() + 1e-10)
#     return new_embeddings, fictional_set
