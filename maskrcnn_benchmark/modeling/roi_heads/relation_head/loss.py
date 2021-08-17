# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        loss_type="CrossEntropy",
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


class WRelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        rel_loss='bce',
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.rel_loss = rel_loss
        self.obj_criterion_loss = nn.CrossEntropyLoss()
        print(self.rel_loss)
        if self.rel_loss == 'bce':
            self.rel_criterion_loss = nn.BCEWithLogitsLoss()
        elif self.rel_loss == "softwt":
            self.rel_criterion_loss = SoftWeightBCE()
        elif self.rel_loss == 'softce':
            self.rel_criterion_loss = CEForSoftLabel()
        elif self.rel_loss == 'attn_bce':
            self.rel_criterion_loss = AttnBCELoss(cfg)
        elif self.rel_loss == "fl":
            self.rel_criterion_loss = BCEFocalLosswithLogits(cfg)
        elif self.rel_loss == "div":
            self.rel_criterion_loss = DivLoss(cfg)

        self.regularization = None
        if cfg.WSUPERVISE.REG.IS_ON:
            self.regularization = Regularization(cfg.WSUPERVISE.REG.WEIGHT)

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, pos_weight=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        refine_obj_logits = refine_logits
        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        # if self.rel_loss == 'softce':
        #     rel_labels = rel_labels/(rel_labels.sum(1)[:, None])

        if self.rel_loss == 'attn_bce':
            loss_relation = self.rel_criterion_loss(relation_logits, rel_labels, pos_weight=pos_weight)
        else:
            loss_relation = self.rel_criterion_loss(relation_logits, rel_labels)


        loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())

        if self.regularization:
            # print(loss_relation, self.regularization(relation_logits, rel_labels))
            loss_reg = self.regularization(relation_logits, rel_labels)
            return loss_relation, loss_reg
        return loss_relation, loss_refine_obj


class SoftWeightBCE(nn.Module):
    """
    Given a normal BCE Loss, apply weights for different classes
    """
    def __init__(self, reduction="mean"):
        super(SoftWeightBCE, self).__init__()
        self.reduction=reduction

    def forward(self, relation_logits, rel_labels, pos_weight=None):
        final_rel_labels = torch.zeros_like(rel_labels)
        final_rel_labels[rel_labels > 0] = 1.
        weights = torch.ones_like(rel_labels)
        weights[rel_labels > 0] = rel_labels[rel_labels > 0]
        loss_mat = F.binary_cross_entropy_with_logits(relation_logits, final_rel_labels.detach(), reduction="none")
        loss_relation = (loss_mat * weights.detach()).mean()
        return loss_relation


class CEForSoftLabel(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(CEForSoftLabel, self).__init__()
        self.reduction=reduction

    def forward(self, input, target, pos_weight=None):
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class Regularization(nn.Module):
    def __init__(self, weight):
        super(Regularization, self).__init__()
        self.softmax = nn.Softmax(1)
        self.weight = weight

    def forward(self, logits, target):
        # regularization
        device = logits.device
        nonzero_idxs = target.nonzero()
        to_fill_in = torch.zeros(logits.size(), dtype=logits.dtype).to(device)
        to_fill_in[:] = -10000
        to_fill_in[nonzero_idxs[:, 0], nonzero_idxs[:, 1]] = logits[nonzero_idxs[:, 0], nonzero_idxs[:, 1]]
        to_fill_in = self.softmax(to_fill_in)
        attn_score = torch.ones(logits.size(), dtype=torch.float32).to(device)
        attn_score[nonzero_idxs[:, 0], nonzero_idxs[:, 1]] = to_fill_in[nonzero_idxs[:, 0], nonzero_idxs[:, 1]]
        reguraization_item = -(attn_score.log()*attn_score).sum()/nonzero_idxs.size(0)
        return reguraization_item*self.weight


class AttnBCELoss(nn.Module):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(self, cfg):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        super(AttnBCELoss, self).__init__()
        if cfg is None:
            self.num_rel = 21
        else:
            self.num_rel = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

    def __call__(self, relation_logits, rel_labels, pos_weight):
        # labels = [torch.zeros(self.num_rel, dtype=torch.float32) for i in range(len(relation_logits))]
        # labels = torch.stack(labels, 0).to(relation_logits.device)
        # for i in range(self.batch_size):
        #     labels[i*self.bag_size: (i+1)*self.bag_size, rel_labels[i]] = 1
        loss = F.binary_cross_entropy_with_logits(relation_logits, rel_labels, reduction="none")
        return (loss*pos_weight.detach()).mean()


class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = cfg.WSUPERVISE.FL.GAMMA
        self.alpha = cfg.WSUPERVISE.FL.ALPHA
        self.beta = cfg.WSUPERVISE.FL.BETA
        self.reduction = reduction
        self.max_val = cfg.WSUPERVISE.FL.MAX_VAL
        self.softmax= nn.Softmax(1)

    def forward(self, logits, target):
        # logits: [N, H, W], target: [N, H, W]
        x = logits
        t = target
        alpha = self.alpha
        gamma = self.gamma
        beta = self.beta
        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (gamma * xt+beta).sigmoid()
        w = alpha * t + (1 - alpha) * (1 - t)
        f_loss = (-w * pt.log() / gamma).mean()
        return f_loss


class DivLoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(DivLoss, self).__init__()

    def forward(self, logits, target):
        logits = logits.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        weight = 1. / (1. + 1e-5 - pt)
        f_loss = (weight * bce_loss).mean()
        return f_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def make_weaksup_relation_loss_evaluator(cfg):
    loss_evaluator = WRelationLossComputation(
        cfg,
        cfg.WSUPERVISE.LOSS_TYPE,
    )

    return loss_evaluator


def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator
