import json
import pickle
import sys
import numpy as np
import torch
ratio = float(sys.argv[1])
alpha = float(sys.argv[2])

print(ratio, alpha)
score = json.load(open("score.json", "r"))
em = pickle.load(open("raw_em_E.pk", "rb"))
clip_data = pickle.load(open("cc_clip_logits.pk", "rb"))

thres = score[int(ratio * len(score))-1]

n = 0
rst = []
for d, clip_d in zip(em, clip_data):
    pairs = d['pairs']
    rel_logits = d['rel_logits']
    possible_rels = d['possible_rels']

    clip_rel_logits = clip_d["rel_logits"]
    clip_possible_rels = clip_d["possible_rels"]

    rst_pairs = []
    rst_rel_logits = []
    rst_possible_rels = []
    for i, (logit, rels, clip_logit, clip_rels) in enumerate(zip(rel_logits, possible_rels, clip_rel_logits, clip_possible_rels)):
        s = torch.tensor(logit).softmax(0)[-1]
        if s > thres:
            continue
        else:
            clip_logit = 1. if clip_logit is None else clip_logit.squeeze()
            to_use_logit = alpha*torch.tensor(logit[0:-1]).float().softmax(0) + (1-alpha)*torch.tensor(clip_logit).float().softmax(0)
            assert (to_use_logit.sum()- 1.).abs() < 1e-5, str(torch.tensor(logit[0:-1]).float().softmax(0)) +" "+ str(torch.tensor(clip_logit).float().softmax(0))
            # retain
            rst_pairs.append(pairs[i])
            rst_rel_logits.append(to_use_logit.numpy())
            rst_possible_rels.append(rels[:-1])
    d['pairs'] = np.asarray(rst_pairs)
    d['rel_logits'] = rst_rel_logits
    d['possible_rels'] = rst_possible_rels
    if rst_rel_logits == []:
        n += 1
        rst.append(None)
    else:
        rst.append(d)
# pickle.dump(rst, open("em_E.pk_alpha"+str(round(alpha, 2)), "wb"))
pickle.dump(rst, open("em_E.pk", "wb"))
print(n)
