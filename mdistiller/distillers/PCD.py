import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def PCD_loss(logits_student_in, logits_teacher_in, temperature, alpha, logit_stand, steps):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    _, num_classes = logits_teacher.shape

    steps = int(steps)
    loss_stol = 0.0
    loss_ltos = 0.0
    for i in range(steps, 0, -1):
        block_size = num_classes 

        for j in range(i):  
            start_idx = j * block_size
            end_idx = (j + 1) * block_size if j != i -1 else num_classes 
            mask = _get_block_mask(logits_teacher, start_idx, end_idx)
            mask = mask.to(logits_student.device)
            assert mask.shape == logits_student.shape

            
            masked_stol_student = (logits_student / temperature).masked_fill(~mask, -1e9)
            log_stol_s = F.log_softmax(masked_stol_student, dim=1)
            masked_stol_teacher = (logits_teacher / temperature).masked_fill(~mask, -1e9)
            pred_stol_t = F.softmax(masked_stol_teacher, dim=1)

            cosine_similarity_stol = F.cosine_similarity(pred_stol_t, log_stol_s, dim=1)
            scale_stol_factor = 1.0 - cosine_similarity_stol

            loss_stol += F.kl_div(log_stol_s, pred_stol_t, reduction='batchmean') * (temperature**2) * scale_stol_factor
    
    for i in range(1, steps + 1):
        block_size = num_classes 

        for j in range(i):
            start_idx = j * block_size
            end_idx = (j + 1) * block_size if j != i - 1 else num_classes

            mask = _get_block_mask(logits_teacher, start_idx, end_idx)
            mask = mask.to(logits_student.device)
            assert mask.shape == logits_student.shape

            masked_ltos_student = (logits_student / temperature).masked_fill(~mask, -1e9)
            log_ltos_s = F.log_softmax(masked_ltos_student, dim=1)
            masked_ltos_teacher = (logits_teacher / temperature).masked_fill(~mask, -1e9)
            pred_ltos_t = F.softmax(masked_ltos_teacher, dim=1)

            cosine_similarity_ltos = F.cosine_similarity(pred_ltos_t, log_ltos_s, dim=1)
            scale_ltos_factor = 1.0 - cosine_similarity_ltos

            loss_ltos += F.kl_div(log_ltos_s, pred_ltos_t, reduction='batchmean') * (temperature**2) * scale_ltos_factor        
    
    return alpha * (loss_stol + loss_ltos)


def _get_block_mask(logits, start_idx, end_idx):
    assert logits.dim() == 2
    batch_size, num_classes = logits.shape

    mask = torch.zeros((batch_size, num_classes), dtype=torch.bool)
    mask[:, start_idx:end_idx] = 1  
    
    return mask


class PCD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(PCD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.PCD.CE_WEIGHT
        self.temperature = cfg.PCD.T
        self.warmup = cfg.PCD.WARMUP
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.steps = cfg.PCD.STEPS
        self.alpha = cfg.PCD.ALPHA
    

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_pcd = min(kwargs["epoch"] / self.warmup, 1.0) * PCD_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            self.alpha,
            self.logit_stand,
            self.steps,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_pcd,
        }
        return logits_student, losses_dict