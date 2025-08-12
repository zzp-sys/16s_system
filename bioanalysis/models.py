from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os
import json

# 预测结果模型 - 匹配现有数据库结构
class PredictionResult(models.Model):
    PREDICTION_TYPES = [
        ('autism_risk', '自闭症风险预测'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='用户')
    sample_id = models.IntegerField(blank=True, null=True, verbose_name='样本ID')  # 对应数据库中的 sample_id
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES, default='autism_risk', verbose_name='预测类型')
    
    # 使用现有数据库字段
    confidence_score = models.FloatField(default=0.0, verbose_name='置信度分数')  # 对应数据库中的 confidence_score
    model_version = models.CharField(max_length=50, default='v1.0', verbose_name='模型版本')
    input_data = models.TextField(default='', verbose_name='输入数据')
    prediction_result = models.TextField(default='{}', verbose_name='预测结果')  # 存储JSON字符串
    
    # 时间信息
    created_time = models.DateTimeField(default=timezone.now, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '预测结果'
        verbose_name_plural = '预测结果'
        ordering = ['-created_time']
        # 添加索引优化查询性能
        indexes = [
            models.Index(fields=['user', 'sample_id']),
            models.Index(fields=['created_time']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - 样本{self.sample_id or '未知'} - {self.created_time}"
    
    def get_confidence(self):
        """获取置信度"""
        return self.confidence_score
    
    def get_prediction_score(self):
        """获取预测分数"""
        return self.confidence_score
    
    def get_detailed_results(self):
        """解析预测结果JSON"""
        try:
            return json.loads(self.prediction_result)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def get_risk_level(self):
        """从预测结果中获取风险等级"""
        results = self.get_detailed_results()
        return results.get('risk_level', '未知')
    
    def get_sample_name(self):
        """从预测结果中获取样本名称"""
        results = self.get_detailed_results()
        return results.get('sample_name', f'样本{self.sample_id}')

class DiseasePredictionResult(models.Model):
    DISEASE_CHOICES = [
        ('T2D', '2型糖尿病'),
        ('CONST', '便秘'),
        ('IBD', '肠道疾病'),
        ('PARK', '帕金森病'),
        ('NAFLD', '非酒精性脂肪肝链'),
        ('OBES', '肥胖'),
        ('ALZ', '阿尔兹海默病'),
        ('COG', '认知功能障碍'),
        ('MIG', '偏头痛疾病'),
        ('IBS', '肠易激综合征'),
        ('DIAR', '腹泻'),
        ('AUTO', '自身免疫性疾病'),
        ('ANOR', '厌食'),
        ('COLC', '结肠性肿瘤'),
        ('IBD', '炎症性肠病'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    sample_name = models.CharField(max_length=255)
    prediction_time = models.DateTimeField(default=timezone.now)
    prediction_results = models.TextField()  # JSON格式存储所有疾病的预测概率

    def set_predictions(self, predictions_dict):
        """设置预测结果"""
        self.prediction_results = json.dumps(predictions_dict)

    def get_predictions(self):
        """获取预测结果"""
        try:
            return json.loads(self.prediction_results)
        except:
            return {}

    def get_highest_risk_disease(self):
        """获取风险最高的疾病"""
        predictions = self.get_predictions()
        if predictions:
            max_risk_disease = max(predictions.items(), key=lambda x: x[1])
            return {
                'disease': next(name for code, name in self.DISEASE_CHOICES if code == max_risk_disease[0]),
                'probability': max_risk_disease[1]
            }
        return None

    class Meta:
        ordering = ['-prediction_time']

# 用户操作日志模型 - 简化版，去掉不必要的关联
class UserActivityLog(models.Model):
    ACTION_TYPES = [
        ('login', '登录'),
        ('logout', '登出'),
        ('upload', '上传文件'),
        ('analysis', '开始分析'),
        ('prediction_start', '开始预测'),
        ('view_result', '查看结果'),
        ('error', '错误'),
        ('delete', '删除数据'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='用户')
    action_type = models.CharField(max_length=20, choices=ACTION_TYPES, verbose_name='操作类型')
    description = models.CharField(max_length=500, verbose_name='操作描述')
    ip_address = models.GenericIPAddressField(verbose_name='IP地址')
    user_agent = models.TextField(verbose_name='用户代理')
    timestamp = models.DateTimeField(default=timezone.now, verbose_name='操作时间')
    
    class Meta:
        verbose_name = '用户操作日志'
        verbose_name_plural = '用户操作日志'
        ordering = ['-timestamp']
        # 添加索引优化查询性能
        indexes = [
            models.Index(fields=['user', 'action_type']),
            models.Index(fields=['timestamp']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.get_action_type_display()} - {self.timestamp}"
