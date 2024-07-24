import torch.nn as nn


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(soft_student_outputs, soft_teacher_outputs):
    return 1 - pearson_correlation(soft_student_outputs, soft_teacher_outputs).mean()


def intra_class_relation(soft_student_outputs, soft_teacher_outputs):
    return inter_class_relation(soft_student_outputs.transpose(0, 1), soft_teacher_outputs.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, temp=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.temp = temp

    def forward(self, student_preds, teacher_preds, **kwargs):
        soft_student_outputs = (student_preds / self.temp).softmax(dim=1)
        soft_teacher_outputs = (teacher_preds / self.temp).softmax(dim=1)
        inter_loss = self.temp ** 2 * inter_class_relation(soft_student_outputs, soft_teacher_outputs)
        intra_loss = self.temp ** 2 * intra_class_relation(soft_student_outputs, soft_teacher_outputs)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

