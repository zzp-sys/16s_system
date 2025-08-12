from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import (
    PredictionResult, UserActivityLog, DiseasePredictionResult
)

# 用户活动日志管理
@admin.register(UserActivityLog)
class UserActivityLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'action_type', 'description', 'ip_address', 'timestamp']
    list_filter = ['action_type', 'timestamp', 'user']
    search_fields = ['user__username', 'description', 'ip_address']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'
    ordering = ['-timestamp']
    
    # 分页设置
    list_per_page = 50
    
    # 字段分组
    fieldsets = (
        ('基本信息', {
            'fields': ('user', 'action_type', 'description')
        }),
        ('技术信息', {
            'fields': ('ip_address', 'user_agent', 'timestamp'),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        # 禁止手动添加日志
        return False
    
    def has_change_permission(self, request, obj=None):
        # 禁止修改日志
        return False

# 预测结果管理
@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ['get_sample_name', 'user', 'prediction_type', 'confidence_score', 'get_risk_level', 'created_time']
    list_filter = ['prediction_type', 'created_time', 'user']
    search_fields = ['user__username', 'input_data']
    readonly_fields = ['created_time']
    date_hierarchy = 'created_time'
    ordering = ['-created_time']
    
    fieldsets = (
        ('基本信息', {
            'fields': ('user', 'sample_id', 'prediction_type')
        }),
        ('预测结果', {
            'fields': ('confidence_score', 'model_version')
        }),
        ('详细信息', {
            'fields': ('input_data', 'prediction_result'),
            'classes': ('collapse',)
        }),
        ('时间信息', {
            'fields': ('created_time',)
        }),
    )
    
    def get_sample_name(self, obj):
        """获取样本名称"""
        return obj.get_sample_name()
    get_sample_name.short_description = '样本名称'
    
    def get_risk_level(self, obj):
        """获取风险等级"""
        return obj.get_risk_level()
    get_risk_level.short_description = '风险等级'

# 疾病预测结果管理
@admin.register(DiseasePredictionResult)
class DiseasePredictionResultAdmin(admin.ModelAdmin):
    list_display = ['user', 'sample_name', 'prediction_time', 'disease_count']
    list_filter = ['prediction_time']
    search_fields = ['user__username', 'sample_name']
    readonly_fields = ['prediction_time']
    date_hierarchy = 'prediction_time'
    
    def disease_count(self, obj):
        """显示预测的疾病数量"""
        predictions = obj.get_predictions()
        return len(predictions) if predictions else 0
    disease_count.short_description = '疾病数量'

# 自定义admin站点信息
admin.site.site_header = '16S微生物分析系统管理'
admin.site.site_title = '16S系统管理'
admin.site.index_title = '欢迎使用16S微生物分析系统管理后台'
