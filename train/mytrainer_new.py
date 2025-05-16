import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import Trainer

class SparsityAdaptiveKD(nn.Module):
    def __init__(self, base_temp=1.0, max_temp=5.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(base_temp))
        self.max_temp = max_temp

    def adjust_temperature(self, sparsity):
        # 确保 sparsity 是一个张量
        if not isinstance(sparsity, torch.Tensor):
            sparsity = torch.tensor(sparsity)
        
        # 计算新的温度值
        new_temp = 0.5 + sparsity * 2.0
        
        # 确保 new_temp 是一个张量
        if not isinstance(new_temp, torch.Tensor):
            new_temp = torch.tensor(new_temp)
        
        # 使用 torch.clamp 限制温度范围
        self.temperature.data = torch.clamp(new_temp, 0.1, self.max_temp)

class KDTrainer(Trainer):
    def __init__(self, teacher_model, reg_lambda=0.3, max_temp=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.ce_loss = CrossEntropyLoss()
        self.kd_module = SparsityAdaptiveKD(max_temp=max_temp)
        self.reg_lambda = reg_lambda
        self.grad_norm_buffer = []
        model = self.model
        if hasattr(model, 'predictor') and model.predictor is not None:
            self.sparsity = model.predictor.get_sparsity()
        else :
            self.sparsity = 0
        self.kd_module.adjust_temperature(self.sparsity)

    def compute_sparsity_weights(self, sparsity):
        """稀疏自适应权重计算（融合网页1的SWAD权重平均思想）"""
        alpha = torch.sigmoid(torch.tensor(sparsity * 10.0))
        beta = 1.0 - alpha
        return alpha.item(), beta.item()

    def activation_regularization(self, activations):
        """激活稀疏正则化（参考网页4的PLUG模型稀疏策略）"""
        active_ratio = torch.mean((activations > 0).float())
        target_sparsity = 0.7  # 可配置参数
        return F.mse_loss(active_ratio, torch.tensor(target_sparsity))

    def gradient_normalization(self, model):
        params = [p for p in model.parameters() if p.grad is not None]
        if len(params) > 0:
            grad_norms = torch.stack([p.grad.detach().data.norm(2) for p in params])
            self.grad_norm_buffer.append(grad_norms.mean().item())
            if len(self.grad_norm_buffer) > 10:
                self.grad_norm_buffer.pop(0)
            avg_grad_norm = sum(self.grad_norm_buffer)/len(self.grad_norm_buffer)
            for p in params:
                p.grad.data.div_(avg_grad_norm + 1e-8)

    def forward_kl_loss(self, labels, student_logits, teacher_logits):
        student_log_prob = F.log_softmax(student_logits/self.kd_module.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits/self.kd_module.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(-1)
        mask = (labels != -100)
        return (kl_loss * mask).sum() / mask.sum()
    

    def compute_loss(self, model, inputs, return_outputs=False):

        # 教师模型推理
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits

        # 学生模型推理
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # 损失计算
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), 
                              inputs['labels'].view(-1))
        kd_loss = self.forward_kl_loss(inputs['labels'], student_logits, teacher_logits)
        
        # 动态权重分配（参考网页5的PromptMM加权策略）
        alpha, beta = self.compute_sparsity_weights(self.sparsity)
        print('ce_loss: ', ce_loss.item(), ' kd_loss: ', kd_loss.item(), ' alpha: ', alpha, ' beta: ', beta, 'sparsity: ', self.sparsity)
        total_loss = (alpha * ce_loss) + (beta * kd_loss)

        # 梯度归一化处理
        if total_loss.requires_grad:
            total_loss.backward()
            self.gradient_normalization(model)

        return (total_loss, student_outputs) if return_outputs else total_loss