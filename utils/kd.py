import torch.nn as nn
import torch.nn.functional as F

# loss = alpha * hard_loss + (1-alpha) * kd_loss，此处是单单的kd_loss
class KLDiv(nn.Module):
    def __init__(self, temp=1.0):
        super(KLDiv, self).__init__()
        self.temp = temp

    def forward(self, student_preds, teacher_preds, **kwargs):
        soft_student_outputs = F.log_softmax(student_preds / self.temp, dim=1)
        soft_teacher_outputs = F.softmax(teacher_preds / self.temp, dim=1)
        kd_loss = F.kl_div(soft_student_outputs, soft_teacher_outputs, reduction="none").sum(1).mean()
        kd_loss *= self.temp ** 2
        return kd_loss

