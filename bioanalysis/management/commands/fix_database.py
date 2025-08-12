from django.core.management.base import BaseCommand
from django.db import connection
import sqlite3
import os

class Command(BaseCommand):
    help = '修复数据库结构问题'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('开始修复数据库结构...'))
        
        try:
            # 检查数据库文件是否存在
            db_path = 'db.sqlite3'
            if not os.path.exists(db_path):
                self.stdout.write(self.style.ERROR(f'数据库文件不存在: {db_path}'))
                return
            
            # 连接数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 检查表结构
            self.stdout.write('检查 bioanalysis_predictionresult 表结构...')
            cursor.execute("PRAGMA table_info(bioanalysis_predictionresult)")
            columns = cursor.fetchall()
            
            if not columns:
                self.stdout.write(self.style.ERROR('表不存在，需要运行迁移'))
                return
            
            self.stdout.write('当前表结构:')
            column_names = []
            for col in columns:
                self.stdout.write(f'  - {col[1]} ({col[2]})')
                column_names.append(col[1])
            
            # 检查是否缺少 analysis_task_id 字段
            if 'analysis_task_id' not in column_names:
                self.stdout.write('缺少 analysis_task_id 字段，正在添加...')
                try:
                    cursor.execute("""
                        ALTER TABLE bioanalysis_predictionresult 
                        ADD COLUMN analysis_task_id INTEGER NULL
                    """)
                    self.stdout.write(self.style.SUCCESS('✓ analysis_task_id 字段已添加'))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'添加字段失败: {e}'))
            else:
                self.stdout.write(self.style.SUCCESS('✓ analysis_task_id 字段已存在'))
            
            # 提交更改
            conn.commit()
            conn.close()
            
            self.stdout.write(self.style.SUCCESS('数据库修复完成'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'修复数据库时出错: {e}'))
            
        # 现在尝试更新视图以重新启用预测结果查询
        self.stdout.write('更新视图以重新启用预测结果查询...')
        try:
            from bioanalysis.models import PredictionResult
            from django.contrib.auth.models import User
            
            # 测试查询
            user = User.objects.first()
            if user:
                prediction_results = PredictionResult.objects.filter(user=user)
                count = prediction_results.count()
                self.stdout.write(self.style.SUCCESS(f'测试查询成功，找到 {count} 个预测结果'))
            else:
                self.stdout.write(self.style.WARNING('没有找到用户进行测试'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'测试查询失败: {e}')) 