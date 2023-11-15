import torch.nn as nn
import torch
import torch.nn.functional as F

@LOSSES_REGISTRY.register('equalized_focal_loss')
class EqualizedFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 name='equalized_focal_loss',
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=1204,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=8.0,
                 fpn_levels=5):
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

        # Focal Loss的超参数
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # ignore bg class and ignore idx
        self.num_classes = num_classes - 1

        # EFL损失函数的超参数
        self.scale_factor = scale_factor
        # 初始化正负样本的梯度变量
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # 初始化正负样本变量
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels

        logger.info(
            "build EqualizedFocalLoss, focal_alpha: {focal_alpha}, focal_gamma: {focal_gamma},scale_factor: {scale_factor}")

    def forward(self, input, target, reduction, normalizer=None):
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c)
        self.target = target.reshape(-1)
        self.n_i, _ = self.input.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, 1:]

        expand_target = expand_label(self.input, self.target)
        sample_mask = (self.target != self.ignore_index)

        inputs = self.input[sample_mask]
        targets = expand_target[sample_mask]
        self.cache_mask = sample_mask
        self.cache_target = expand_target

        pred = torch.sigmoid(inputs)
        pred_t = pred * targets + (1 - pred) * (1 - targets)
        # map_val为：1-g^j
        map_val = 1 - self.pos_neg.detach()
        # dy_gamma为：gamma^j
        dy_gamma = self.focal_gamma + self.scale_factor * map_val

        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)[sample_mask]

        # weighting factor
        wf = ff / self.focal_gamma

        # ce_loss
        ce_loss = -torch.log(pred_t)
        cls_loss = ce_loss * torch.pow((1 - pred_t), ff.detach()) * wf.detach()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            cls_loss = alpha_t * cls_loss

        if normalizer is None:
            normalizer = 1.0

        return _reduce(cls_loss, reduction, normalizer=normalizer)

    # 收集梯度，用于梯度引导的机制
    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        self.grad_buffer.append(grad_in.detach().permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes))
        if len(self.grad_buffer) == self.fpn_levels:
            target = self.cache_target[self.cache_mask]
            grad = torch.cat(self.grad_buffer[::-1], dim=1).reshape(-1, self.num_classes)

            grad = torch.abs(grad)[self.cache_mask]
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

            allreduce(pos_grad)
            allreduce(neg_grad)
            # 正样本的梯度
            self.pos_grad += pos_grad
            # 负样本的梯度
            self.neg_grad += neg_grad
            # self.pos_neg=g_j:表示第j类正样本与负样本的累积梯度比
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)
            self.grad_buffer = []
