from django.shortcuts import render, redirect
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .forms import FileUploadForm
from .utils import run_fastp, generate_manifest, run_qiime2_analysis, pred_model, predict_multiple_diseases
import os
import time
import subprocess
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import pandas as pd
import joblib
from biom import load_table
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserActivityLog, PredictionResult
from django.db.models import Count
from django.utils import timezone
from datetime import datetime, timedelta
import json
from django.db import connection
import hashlib
from .models import DiseasePredictionResult

def get_client_ip(request):
    """获取客户端IP地址"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def log_user_activity(user, action_type, description, request, sample_name=None):
    """记录用户活动"""
    try:
        UserActivityLog.objects.create(
            user=user,
            action_type=action_type,
            description=description,
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')
        )
    except Exception as e:
        print(f"Failed to log user activity: {e}")

MEDIA = 'D:/A_project/code_system/media'




# 获取日志 - 需要登录
@login_required
def get_log(request):
    log_path = 'media/analysis.log'
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        content = ''
    return JsonResponse({'log': content})

# 清空日志 - 需要登录
@login_required
def clear_log(request):
    if request.method == 'POST':
        log_path = 'media/analysis.log'
        try:
            # 清空日志文件
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('')
            
            # 终止正在运行的Docker容器和相关进程
            try:
                import subprocess
                import platform
                
                # 尝试停止特定的 Docker 容器（只停止分析相关的容器）
                try:
                    # 获取正在运行的 Docker 容器，只查找分析相关的容器
                    result = subprocess.run(['docker', 'ps', '--format', '{{.ID}} {{.Image}}'], 
                                         capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 2:
                                    container_id = parts[0]
                                    image_name = parts[1]
                                    # 只停止分析相关的容器
                                    if any(keyword in image_name.lower() for keyword in ['fastp', 'qiime', 'amplicon']):
                                        try:
                                            # 首先尝试优雅停止（给容器时间保存状态）
                                            print(f"Attempting to stop container {container_id}")
                                            stop_result = subprocess.run(['docker', 'stop', container_id], 
                                                                       capture_output=True, text=True, timeout=15)
                                            
                                            if stop_result.returncode != 0:
                                                print(f"Graceful stop failed for {container_id}, trying force kill")
                                                # 如果优雅停止失败，强制杀死容器
                                                subprocess.run(['docker', 'kill', container_id], 
                                                             capture_output=True, text=True, timeout=10)
                                            else:
                                                print(f"Successfully stopped container {container_id}")
                                                
                                        except subprocess.TimeoutExpired:
                                            print(f"Timeout stopping container {container_id}, forcing kill")
                                            # 超时后强制杀死容器
                                            try:
                                                subprocess.run(['docker', 'kill', container_id], 
                                                             capture_output=True, text=True, timeout=10)
                                            except Exception as kill_error:
                                                print(f"Failed to kill container {container_id}: {kill_error}")
                                        except Exception as stop_error:
                                            print(f"Error stopping container {container_id}: {stop_error}")
                except Exception as e:
                    print(f"Docker stop error: {e}")
                
                # 根据操作系统终止特定的分析进程
                if platform.system() == 'Windows':
                    # Windows 系统 - 只终止分析相关的进程
                    try:
                        # 只终止分析相关的进程，不包括 python.exe（避免终止 Django 服务器）
                        processes_to_kill = ['fastp.exe', 'qiime.exe']
                        for process in processes_to_kill:
                            try:
                                # 检查进程是否存在
                                check_result = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {process}'], 
                                                           capture_output=True, text=True, timeout=5)
                                if process in check_result.stdout:
                                    # 进程存在，终止它
                                    subprocess.run(['taskkill', '/F', '/IM', process], 
                                                 capture_output=True, timeout=5)
                            except Exception as e:
                                print(f"Error killing process {process}: {e}")
                    except Exception as e:
                        print(f"Windows process cleanup error: {e}")
                else:
                    # Linux/Unix系统 - 终止相关进程
                    try:
                        # 使用pkill终止相关进程
                        processes_to_kill = ['fastp', 'qiime', 'docker']
                        for process in processes_to_kill:
                            try:
                                subprocess.run(['pkill', '-f', process], 
                                             capture_output=True, timeout=5)
                            except Exception as e:
                                print(f"Error killing process {process}: {e}")
                    except Exception as e:
                        print(f"Linux process cleanup error: {e}")
                        
            except Exception as e:
                print(f"Process cleanup error: {e}")
            
            return JsonResponse({'status': 'success', 'message': '日志已清空，相关进程已终止'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


# 获取可用样本 - 需要登录
@login_required
def get_available_samples(request):
    """获取当前用户可用于预测的样本列表"""
    try:
        # 使用统一的样本检查逻辑
        user_samples = get_user_samples_with_status(request.user)
        
        # 只返回已完成分析的样本
        available_samples = [
            {
                'name': sample['name'],
                'status': sample['status'],
                'user': request.user.username
            }
            for sample in user_samples if sample['is_completed']
        ]
        
        return JsonResponse({'samples': available_samples})
        
    except Exception as e:
        return JsonResponse({'samples': [], 'error': str(e)})

def get_user_samples_with_status(user):
    """获取用户样本列表及其状态（统一的样本检查逻辑）"""
    user_samples = []
    media_dir = 'media'
    
    try:
        # 获取用户的样本名称集合
        user_sample_names = set()
        
        # 方法1：从UserActivityLog中获取用户上传的样本
        upload_logs = UserActivityLog.objects.filter(
            user=user,
            action_type='upload'
        ).values_list('description', flat=True)
        
        for log_desc in upload_logs:
            # 从描述中提取样本名
            if '上传样本文件:' in log_desc:
                sample_name = log_desc.split('上传样本文件:')[1].split('(')[0].strip()
                user_sample_names.add(sample_name)
        
        # 方法2：从分析日志中获取用户样本
        if os.path.exists(media_dir):
            log_file = os.path.join(media_dir, 'analysis.log')
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        current_user = None
                        current_sample = None
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith('[USER]'):
                                current_user = line.replace('[USER]', '').strip()
                            elif line.startswith('[SAMPLE]'):
                                current_sample = line.replace('[SAMPLE]', '').strip()
                            elif line.startswith('[STAGE] done') and current_user == user.username:
                                if current_sample:
                                    user_sample_names.add(current_sample)
                except Exception as e:
                    print(f"Error reading log file: {e}")
        
        # 方法3：如果是管理员或者没有找到样本，检查media目录
        if user.is_superuser or not user_sample_names:
            if os.path.exists(media_dir):
                for item in os.listdir(media_dir):
                    item_path = os.path.join(media_dir, item)
                    if os.path.isdir(item_path) and item not in ['analysis.log', 'Biom']:
                        if user.is_superuser:
                            user_sample_names.add(item)
                        else:
                            # 对于非管理员，只有当样本有完成标记时才计入
                            required_files = [
                                'Biom/OTU_table.tsv',
                                'Biom/feature-table.biom'
                            ]
                            has_all_files = True
                            for file_name in required_files:
                                file_path = os.path.join(item_path, file_name)
                                if not os.path.exists(file_path):
                                    has_all_files = False
                                    break
                            if has_all_files:
                                user_sample_names.add(item)
        
        # 检查每个样本的状态
        for sample_name in user_sample_names:
            sample_path = os.path.join(media_dir, sample_name)
            if os.path.isdir(sample_path):
                # 检查分析结果文件是否存在
                required_files = [
                    'Biom/OTU_table.tsv',
                    'Biom/feature-table.biom'
                ]
                
                has_all_files = True
                for file_name in required_files:
                    file_path = os.path.join(sample_path, file_name)
                    if not os.path.exists(file_path):
                        has_all_files = False
                        break
                
                status = '已完成分析' if has_all_files else '分析中'
                
                sample_info = {
                    'name': sample_name,
                    'status': status,
                    'path': sample_path,
                    'is_completed': has_all_files
                }
                user_samples.append(sample_info)
        
        return user_samples
                
    except Exception as e:
        print(f"Error getting user samples with status: {e}")
        return []

def get_user_samples(user):
    """获取当前用户的样本列表（保持向后兼容）"""
    user_samples_with_status = get_user_samples_with_status(user)
    return [sample['name'] for sample in user_samples_with_status]

@login_required
def integrated_analysis(request):
    """集成分析和预测功能"""
    if request.method == 'POST':
        # 检查是否是文件上传请求
        if 'file' in request.FILES:
            try:
                # 处理文件上传和分析
                uploaded_files = request.FILES.getlist('file')
                
                if not uploaded_files:
                    return JsonResponse({'status': 'error', 'message': '没有选择文件'})
                
                # 验证文件类型
                valid_extensions = ('.fastq', '.fq', '.fastq.gz', '.fq.gz')
                fastq_files = []
                
                for file in uploaded_files:
                    if file.name.lower().endswith(valid_extensions):
                        fastq_files.append(file)
                    else:
                        return JsonResponse({'status': 'error', 'message': f'不支持的文件类型: {file.name}'})
                
                if len(fastq_files) % 2 != 0:
                    return JsonResponse({'status': 'error', 'message': '请选择成对的FASTQ文件（_1.fastq和_2.fastq）'})
                
                # 创建样本目录
                sample_name = None
                sample_pairs = {}
                
                # 分组配对文件
                for file in fastq_files:
                    base_name = file.name.replace('_1.fastq', '').replace('_2.fastq', '').replace('.fastq', '')
                    if base_name not in sample_pairs:
                        sample_pairs[base_name] = {}
                    
                    if '_1.fastq' in file.name or file.name.endswith('_1.fq'):
                        sample_pairs[base_name]['R1'] = file
                        sample_name = base_name
                    elif '_2.fastq' in file.name or file.name.endswith('_2.fq'):
                        sample_pairs[base_name]['R2'] = file
                        sample_name = base_name
                    else:
                        # 如果文件名不包含_1或_2，假设是单端文件或需要用户重命名
                        return JsonResponse({'status': 'error', 'message': f'文件命名不符合要求，请确保文件名包含_1和_2标识: {file.name}'})
                
                # 验证每个样本都有配对文件
                for base_name, files in sample_pairs.items():
                    if 'R1' not in files or 'R2' not in files:
                        return JsonResponse({'status': 'error', 'message': f'样本 {base_name} 缺少配对文件'})
                
                # 选择第一个样本作为主要样本（如果有多个样本，只处理第一个）
                if len(sample_pairs) > 1:
                    # 如果有多个样本，提示用户
                    sample_name = list(sample_pairs.keys())[0]
                    return JsonResponse({'status': 'error', 'message': f'当前只支持单个样本分析，检测到多个样本。将使用第一个样本: {sample_name}'})
                
                sample_name = list(sample_pairs.keys())[0]
                files_pair = sample_pairs[sample_name]
                
                # 创建样本目录
                sample_dir = os.path.join(MEDIA, sample_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                # 保存上传的文件
                try:
                    # 保存R1文件
                    r1_path = os.path.join(sample_dir, f'{sample_name}_1.fastq')
                    with open(r1_path, 'wb+') as destination:
                        for chunk in files_pair['R1'].chunks():
                            destination.write(chunk)
                    
                    # 保存R2文件
                    r2_path = os.path.join(sample_dir, f'{sample_name}_2.fastq')
                    with open(r2_path, 'wb+') as destination:
                        for chunk in files_pair['R2'].chunks():
                            destination.write(chunk)
                    
                    # 记录用户活动
                    log_user_activity(
                        user=request.user,
                        action_type='upload',
                        description=f'上传样本文件: {sample_name} ({len(fastq_files)}个文件)',
                        request=request
                    )
                    
                    # 初始化日志文件
                    log_file = 'media/analysis.log'
                    with open(log_file, 'w', encoding='utf-8') as lf:
                        lf.write(f'[USER] {request.user.username}\n')
                        lf.write(f'[SAMPLE] {sample_name}\n')
                        lf.write(f'[FILES] {sample_name}_1.fastq|{sample_name}_2.fastq\n')
                        lf.write('[STAGE] fastp\n')
                    
                    # 启动后台分析任务
                    import threading
                    def run_analysis():
                        try:
                            # 转换sample_dir为Docker兼容路径
                            sample_dir_abs = os.path.abspath(sample_dir)
                            if os.name == 'nt':  # Windows系统
                                sample_dir_docker = sample_dir_abs.replace('\\', '/')
                                if len(sample_dir_docker) >= 2 and sample_dir_docker[1] == ':':
                                    drive_letter = sample_dir_docker[0].lower()
                                    sample_dir_docker = f'/{drive_letter}{sample_dir_docker[2:]}'
                            else:
                                sample_dir_docker = sample_dir_abs
                            
                            # 运行FastP质控 - 使用Docker兼容路径
                            if run_fastp(sample_name, sample_dir_docker, log_file):
                                # 生成manifest文件
                                manifest_file = os.path.join(sample_dir, 'manifest.txt')
                                generate_manifest(sample_dir, manifest_file)
                                
                                # 运行QIIME2分析 - 使用Docker兼容路径
                                with open(log_file, 'a', encoding='utf-8') as lf:
                                    lf.write('[STAGE] qiime2\n')
                                
                                if run_qiime2_analysis(sample_dir_docker, log_file):
                                    with open(log_file, 'a', encoding='utf-8') as lf:
                                        lf.write('[STAGE] done\n')
                                        lf.write('[STATUS] completed\n')
                                else:
                                    with open(log_file, 'a', encoding='utf-8') as lf:
                                        lf.write('[STAGE] failed\n')
                                        lf.write('[STATUS] error\n')
                            else:
                                with open(log_file, 'a', encoding='utf-8') as lf:
                                    lf.write('[STAGE] failed\n')
                                    lf.write('[STATUS] error\n')
                        except Exception as e:
                            with open(log_file, 'a', encoding='utf-8') as lf:
                                lf.write(f'[ERROR] {str(e)}\n')
                                lf.write('[STATUS] error\n')
                    
                    # 启动分析线程
                    analysis_thread = threading.Thread(target=run_analysis)
                    analysis_thread.daemon = True
                    analysis_thread.start()
                    
                    return JsonResponse({
                        'status': 'success', 
                        'message': f'文件上传成功，开始分析样本: {sample_name}',
                        'sample_name': sample_name
                    })
                    
                except Exception as e:
                    return JsonResponse({'status': 'error', 'message': f'文件保存失败: {str(e)}'})
                    
            except Exception as e:
                return JsonResponse({'status': 'error', 'message': f'上传处理失败: {str(e)}'})
        
        # 处理预测请求
        sample_name = request.POST.get('sample_name')
        
        if not sample_name:
            return JsonResponse({'success': False, 'error': '未指定样本名称'})
        
        # 检查样本是否存在
        available_samples_response = get_available_samples(request)
        if hasattr(available_samples_response, 'content'):
            import json
            available_samples_data = json.loads(available_samples_response.content)
            available_samples = [sample['name'] for sample in available_samples_data.get('samples', [])]
        else:
            available_samples = []
        
        if sample_name not in available_samples:
            return JsonResponse({'success': False, 'error': '指定的样本不存在'})
        
        # 构建数据目录路径
        data_dir = os.path.join('media', sample_name)
        
        # 检查分析是否已完成
        required_files = ['table.qza', 'taxonomy.qza', 'rooted-tree.qza']
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                return JsonResponse({'success': False, 'error': f'样本 {sample_name} 的分析尚未完成，缺少必要文件: {file}'})
        
        # 记录预测开始活动
        log_user_activity(
            user=request.user,
            action_type='prediction_start',
            description=f'开始自闭症风险预测: {sample_name}',
            request=request
        )

        # 进行预测
        if request.POST.get('action') == 'predict':
            try:
                # 调用预测模型
                pred_result = pred_model(data_dir)
                
                # 检查是否返回False（文件不存在的情况）
                if pred_result is False:
                    return JsonResponse({'success': False, 'error': '预测所需文件不存在，请确保样本已完成分析'})
                
                # 解包预测结果
                y_pred_rpart, y_pred_rf = pred_result
                
                # 计算风险等级 - 仅基于决策树(RPART)模型
                rpart_confidence = float(y_pred_rpart[0].max())
                rpart_prediction = int(y_pred_rpart[0].argmax())
                rpart_control_prob = float(y_pred_rpart[0][0])
                rpart_autism_prob = float(y_pred_rpart[0][1])

                # 同时获取RF模型数据用于展示
                rf_confidence = float(y_pred_rf[0].max())
                rf_prediction = int(y_pred_rf[0].argmax())

                # 风险评估逻辑 - 基于决策树模型的自闭症概率
                # 注意：y_pred_rpart[0][0] 是控制组概率，y_pred_rpart[0][1] 是自闭症概率
                if rpart_autism_prob >= 0.7:  # 自闭症概率 >= 70%
                    risk_level = "高风险"
                elif rpart_autism_prob >= 0.5:  # 自闭症概率 >= 50%
                    risk_level = "中风险"
                else:  # 自闭症概率 < 50%
                    risk_level = "低风险"

                # 准备预测数据
                prediction_data = {
                    'sample_name': sample_name,
                    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'user_id': request.user.id,
                    'username': request.user.username,
                    'risk_level': risk_level,
                    'rf_model': {
                        'probabilities': [round(float(prob), 4) for prob in y_pred_rf[0].tolist()],
                        'prediction': int(y_pred_rf[0].argmax()),
                        'confidence': round(float(y_pred_rf[0].max()), 4),
                        'control_probability': round(float(y_pred_rf[0][0]), 4),
                        'autism_probability': round(float(y_pred_rf[0][1]), 4)
                    },
                    'rpart_model': {
                        'probabilities': [round(float(prob), 4) for prob in y_pred_rpart[0].tolist()],
                        'prediction': int(y_pred_rpart[0].argmax()),
                        'confidence': round(float(y_pred_rpart[0].max()), 4),
                        'control_probability': round(float(y_pred_rpart[0][0]), 4),
                        'autism_probability': round(float(y_pred_rpart[0][1]), 4)
                    },
                    'labels': ['控制组', '自闭症'],
                    'model_info': {
                        'primary_model': 'Decision Tree (RPART)',
                        'secondary_model': 'Random Forest',
                        'description': '基于16S rRNA基因序列的自闭症风险预测模型，风险评估基于决策树模型结果'
                    }
                }
                
                # 保存预测结果到JSON文件
                try:
                    # 确保样本目录存在
                    sample_dir = os.path.join('media', sample_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # 生成JSON文件名（包含时间戳避免覆盖）
                    json_filename = f'prediction_result.json'
                    json_filepath = os.path.join(sample_dir, json_filename)
                    
                    # 保存JSON文件
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(prediction_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"预测结果已保存到: {json_filepath}")
                    
                    # 同时保存一个最新结果的副本（固定文件名，方便读取）
                    latest_json_filepath = os.path.join(sample_dir, 'latest_prediction.json')
                    with open(latest_json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(prediction_data, f, ensure_ascii=False, indent=2)
                    
                except Exception as json_error:
                    print(f"保存JSON文件时出错: {json_error}")
                    # 即使JSON保存失败，也继续执行后续逻辑
                
                # 保存预测结果到数据库
                try:
                    # 检查PredictionResult模型的实际字段
                    cursor = connection.cursor()
                    cursor.execute("PRAGMA table_info(bioanalysis_predictionresult)")
                    columns = cursor.fetchall()
                    available_fields = [col[1] for col in columns]
                    print(f"PredictionResult表的可用字段: {available_fields}")
                    
                    # 根据实际字段创建预测结果
                    prediction_data_to_save = {
                        'user': request.user,
                        'prediction_type': 'autism_risk',
                        'confidence_score': rpart_confidence,  # 使用决策树置信度
                        'model_version': 'v1.0',
                        'input_data': sample_name,
                        'prediction_result': json.dumps(prediction_data, ensure_ascii=False)
                    }
                    
                    # 如果数据库有sample_id字段，生成一个有效的sample_id
                    if 'sample_id' in available_fields:
                        import hashlib
                        sample_hash = hashlib.md5(f"{sample_name}_{request.user.id}".encode()).hexdigest()
                        prediction_data_to_save['sample_id'] = int(sample_hash[:8], 16) % 2147483647
                    
                    prediction_result = PredictionResult.objects.create(**prediction_data_to_save)
                    
                    # print(f"预测结果已保存到数据库，ID: {prediction_result.id}")
                    
                except Exception as save_error:
                    print(f"保存数据库记录时出错: {save_error}")
                    import traceback
                    traceback.print_exc()
                
                # 记录预测完成活动
                log_user_activity(
                    user=request.user,
                    action_type='view_result',
                    description=f'预测分析完成: {sample_name} (决策树置信度: {rpart_confidence:.2f})',
                    request=request
                )
                
                # 格式化预测结果返回给前端
                result = {
                    'success': True,
                    'sample_name': sample_name,
                    'json_file_saved': True,
                    'predictions': {
                        'rpart': {
                            'probabilities': y_pred_rpart[0].tolist(),
                            'prediction': int(y_pred_rpart[0].argmax()),
                            'confidence': float(y_pred_rpart[0].max())
                        },
                        'rf': {
                            'probabilities': y_pred_rf[0].tolist(),
                            'prediction': int(y_pred_rf[0].argmax()),
                            'confidence': float(y_pred_rf[0].max())
                        }
                    }
                }
                return JsonResponse(result)
                
            except Exception as e:
                # 记录预测失败活动
                log_user_activity(
                    user=request.user,
                    action_type='error',
                    description=f'预测分析失败: {sample_name} - {str(e)}',
                    request=request
                )
                return JsonResponse({'success': False, 'error': f'预测过程中出错: {str(e)}'})

        return JsonResponse({'success': False, 'error': '无效的请求'})
    
    # 处理GET请求 - 显示页面
    else:
        try:
            # 获取用户统计数据
            user_stats = get_user_statistics(request.user)
            
            # 获取用户样本列表
            user_samples = get_user_samples_with_status(request.user)
            
            # 只显示最近5个样本
            recent_samples = []
            for sample in user_samples[:5]:
                sample_info = {
                    'name': sample['name'],
                    'status': sample['status'],
                    'path': sample['path'],
                    'created_time': datetime.fromtimestamp(os.path.getctime(sample['path'])) if os.path.exists(sample['path']) else datetime.now()
                }
                recent_samples.append(sample_info)
            
            context = {
                'user_stats': user_stats,
                'user_samples': user_samples,
                'user': request.user
            }
            
            return render(request, 'bioanalysis/integrated_analysis.html', context)
            
        except Exception as e:
            from django.contrib import messages
            messages.error(request, f'加载页面时出错: {str(e)}')
            return redirect('dashboard')

def get_user_statistics(user):
    """获取用户相关的统计数据"""
    try:
        # 使用统一的样本检查逻辑
        user_samples = get_user_samples_with_status(user)
        
        # 统计数据
        total_samples = len(user_samples)
        completed_analyses = len([s for s in user_samples if s['is_completed']])
        
        # 统计预测次数 - 只统计成功的预测
        predictions_made = UserActivityLog.objects.filter(
            user=user,
            action_type='view_result',
            description__contains='预测分析完成'
            ).count()
        
        # 活跃用户数（如果是管理员显示所有活跃用户，否则显示1）
        if user.is_superuser:
            # 管理员可以看到所有活跃用户数（最近30天有活动的用户）
            from datetime import timedelta
            from django.utils import timezone
            thirty_days_ago = timezone.now() - timedelta(days=30)
            active_users = UserActivityLog.objects.filter(
                timestamp__gte=thirty_days_ago
            ).values('user').distinct().count()
            
            # 如果没有活动日志，则显示总用户数
            if active_users == 0:
                from django.contrib.auth.models import User
                active_users = User.objects.filter(is_active=True).count()
        else:
            # 普通用户只显示自己
            active_users = 1
        
        return {
            'total_samples': total_samples,
            'completed_analyses': completed_analyses,
            'predictions_made': predictions_made,
            'active_users': active_users,
        }
        
    except Exception as e:
        print(f"Error calculating user statistics: {e}")
        return {
            'total_samples': 0,
            'completed_analyses': 0,
            'predictions_made': 0,
            'active_users': 1,
        }

# 终止分析进程 - 需要登录
@login_required
def stop_analysis(request):
    """终止正在运行的分析进程"""
    if request.method == 'POST':
        try:
            import subprocess
            import platform
            
            # 清空日志文件并写入终止标记
            log_path = 'media/analysis.log'
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('[INFO] Analysis stopped by user\n')
                f.write('[STAGE] stopped\n')  # 添加终止标记
                f.write('[STATUS] terminated\n')  # 添加状态标记
            
            # 尝试停止特定的 Docker 容器（只停止分析相关的容器）
            try:
                # 获取正在运行的 Docker 容器，只查找分析相关的容器
                result = subprocess.run(['docker', 'ps', '--format', '{{.ID}} {{.Image}}'], 
                                     capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                container_id = parts[0]
                                image_name = parts[1]
                                # 只停止分析相关的容器
                                if any(keyword in image_name.lower() for keyword in ['fastp', 'qiime', 'amplicon']):
                                    try:
                                        # 首先尝试优雅停止（给容器时间保存状态）
                                        print(f"Attempting to stop container {container_id}")
                                        stop_result = subprocess.run(['docker', 'stop', container_id], 
                                                                   capture_output=True, text=True, timeout=15)
                                        
                                        if stop_result.returncode != 0:
                                            print(f"Graceful stop failed for {container_id}, trying force kill")
                                            # 如果优雅停止失败，强制杀死容器
                                            subprocess.run(['docker', 'kill', container_id], 
                                                         capture_output=True, text=True, timeout=10)
                                        else:
                                            print(f"Successfully stopped container {container_id}")
                                            
                                    except subprocess.TimeoutExpired:
                                        print(f"Timeout stopping container {container_id}, forcing kill")
                                        # 超时后强制杀死容器
                                        try:
                                            subprocess.run(['docker', 'kill', container_id], 
                                                         capture_output=True, text=True, timeout=10)
                                        except Exception as kill_error:
                                            print(f"Failed to kill container {container_id}: {kill_error}")
                                    except Exception as stop_error:
                                        print(f"Error stopping container {container_id}: {stop_error}")
            except Exception as e:
                print(f"Docker stop error: {e}")
            
            # 根据操作系统终止特定的分析进程
            if platform.system() == 'Windows':
                # Windows 系统 - 只终止分析相关的进程
                try:
                    # 只终止分析相关的进程，不包括 python.exe（避免终止 Django 服务器）
                    processes_to_kill = ['fastp.exe', 'qiime.exe']
                    for process in processes_to_kill:
                        try:
                            # 检查进程是否存在
                            check_result = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {process}'], 
                                                       capture_output=True, text=True, timeout=5)
                            if process in check_result.stdout:
                                # 进程存在，终止它
                                subprocess.run(['taskkill', '/F', '/IM', process], 
                                             capture_output=True, timeout=5)
                        except Exception as e:
                            print(f"Error killing {process}: {e}")
                except Exception as e:
                    print(f"Windows process termination error: {e}")
            else:
                # Linux/Unix 系统 - 只终止分析相关的进程
                try:
                    # 使用更精确的进程匹配，避免终止 Django 服务器
                    subprocess.run(['pkill', '-f', 'fastp.*docker'], capture_output=True, timeout=5)
                    subprocess.run(['pkill', '-f', 'qiime.*docker'], capture_output=True, timeout=5)
                    # 不终止 docker 进程本身，只终止分析相关的容器
                except Exception as e:
                    print(f"Linux process termination error: {e}")
            
            return JsonResponse({'status': 'success', 'message': '分析进程已终止'})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'终止进程时出错: {str(e)}'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if not username or not password:
            return render(request, 'bioanalysis/register.html', {'error': '用户名和密码不能为空'})
        if User.objects.filter(username=username).exists():
            return render(request, 'bioanalysis/register.html', {'error': '用户名已存在'})
        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        
        # 记录注册活动
        log_user_activity(
            user=user,
            action_type='login',
            description=f'用户注册并登录: {username}',
            request=request
        )
        
        return redirect('integrated_analysis')
    return render(request, 'bioanalysis/register.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            
            # 记录登录活动
            log_user_activity(
                user=user,
                action_type='login',
                description=f'用户登录: {username}',
                request=request
            )
            
            return redirect('integrated_analysis')
        else:
            return render(request, 'bioanalysis/login.html', {'error': '用户名或密码错误'})
    return render(request, 'bioanalysis/login.html')

def logout_view(request):
    if request.user.is_authenticated:
        # 记录登出活动
        log_user_activity(
            user=request.user,
            action_type='logout',
            description=f'用户登出: {request.user.username}',
            request=request
        )
    
    logout(request)
    return redirect('login')

# 用户首页 - 需要登录
@login_required
def dashboard(request):
    """用户首页，显示分析历史和系统状态"""
    try:
        # 获取用户最近的活动记录
        recent_activities = UserActivityLog.objects.filter(user=request.user).order_by('-timestamp')[:10]
        
        # 获取用户活动统计
        activity_stats = UserActivityLog.objects.filter(user=request.user).values('action_type').annotate(count=Count('action_type'))
        
        # 使用统一的样本检查逻辑
        user_samples = get_user_samples_with_status(request.user)
        
        # 只显示最近5个样本
        recent_samples = []
        for sample in user_samples[:5]:
            sample_info = {
                'name': sample['name'],
                'status': sample['status'],
                'path': sample['path'],
                'created_time': datetime.fromtimestamp(os.path.getctime(sample['path'])) if os.path.exists(sample['path']) else datetime.now()
            }
            recent_samples.append(sample_info)
        
        # 系统状态
        system_status = {
            'total_samples': len(user_samples),
            'completed_samples': len([s for s in user_samples if s['is_completed']]),
            'pending_samples': len([s for s in user_samples if not s['is_completed']]),
            'user_activities': len(recent_activities)
        }
        
        context = {
            'recent_activities': recent_activities,
            'activity_stats': activity_stats,
            'samples': recent_samples,
            'system_status': system_status,
            'user': request.user
        }
        
        return render(request, 'bioanalysis/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'加载首页时出错: {str(e)}')
        return render(request, 'bioanalysis/dashboard.html', {'error': str(e)})

@login_required
def get_user_stats_api(request):
    """API端点：获取用户统计数据"""
    if request.method == 'GET':
        try:
            user_stats = get_user_statistics(request.user)
            # 直接返回统计数据，不包装在data字段中
            return JsonResponse(user_stats)
        except Exception as e:
            return JsonResponse({
                'total_samples': 0,
                'completed_analyses': 0,
                'predictions_made': 0,
                'active_users': 1,
                'error': f'获取统计数据时出错: {str(e)}'
            })
    else:
        return JsonResponse({
            'total_samples': 0,
            'completed_analyses': 0,
            'predictions_made': 0,
            'active_users': 1,
            'error': 'Only GET method is allowed'
        })

@login_required
def sample_results(request):
    """查看所有样本的分析和预测结果"""
    
    try:
        # 获取用户样本信息
        user_samples = get_user_samples_with_status(request.user)
        
        # 获取数据库中的预测结果
        prediction_results = []
        try:
            # 直接使用Django ORM查询PredictionResult
            prediction_results = PredictionResult.objects.filter(user=request.user).order_by('-created_time')
            print(f"从数据库加载了 {len(prediction_results)} 个预测结果")
                
        except Exception as db_error:
            print(f"数据库查询错误: {db_error}")
            prediction_results = []
        
        # 获取样本信息 - 不再使用AnalysisTask
        samples_data = []
        for sample in user_samples:
            sample_name = sample['name']
            sample_info = {
                'name': sample_name,
                'status': sample['status'],
                'is_completed': sample['is_completed'],
                'created_time': None,
                'predictions': [],
                'has_16s_report': False,
                'report_path': None,
                'prediction_summary': {
                    'latest_risk_level': '未预测',
                    'latest_confidence': 0,
                    'prediction_count': 0,
                    'latest_prediction_time': None
                },
                'json_predictions': []  # 新增：从JSON文件读取的预测结果
            }
            
            # 获取样本创建时间
            try:
                if os.path.exists(sample['path']):
                    sample_info['created_time'] = datetime.fromtimestamp(os.path.getctime(sample['path']))
            except Exception as e:
                sample_info['created_time'] = datetime.now()
            
            # 读取JSON预测结果
            try:
                sample_dir = os.path.join('media', sample_name)
                if os.path.exists(sample_dir):
                    # 读取最新的预测结果文件
                    json_path = os.path.join(sample_dir, 'prediction_result.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            sample_info['json_predictions'].append({
                                'file_name': 'prediction_result.json',
                                'data': json_data,
                                'is_latest': True
                            })
                    
                    # 为了向后兼容，如果没有prediction_result.json，则尝试读取latest_prediction.json
                    elif os.path.exists(os.path.join(sample_dir, 'latest_prediction.json')):
                        latest_json_path = os.path.join(sample_dir, 'latest_prediction.json')
                        with open(latest_json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            sample_info['json_predictions'].append({
                                'file_name': 'latest_prediction.json',
                                'data': json_data,
                                'is_latest': True
                            })
                    
                    # 按时间排序JSON预测结果（虽然现在通常只有一个文件）
                    sample_info['json_predictions'].sort(
                        key=lambda x: x['data'].get('prediction_time', ''), 
                        reverse=True
                    )
                    
            except Exception as e:
                print(f"Error reading JSON predictions for {sample_name}: {e}")
            
            # 查找相关的预测结果（数据库中的）
            sample_predictions = []
            for pred in prediction_results:
                try:
                    # 从JSON预测结果中获取样本名称
                    pred_detailed_results = pred.get_detailed_results()
                    pred_sample_name = pred_detailed_results.get('sample_name', '')
                    
                    if pred_sample_name == sample_name:
                        pred_info = {
                            'id': pred.id,
                            'prediction_type': getattr(pred, 'prediction_type', 'autism_risk'),
                            'risk_level': pred.get_risk_level(),
                            'confidence': pred.get_confidence(),
                            'prediction_score': pred.get_prediction_score(),
                            'created_time': getattr(pred, 'created_time', datetime.now()),
                            'detailed_results': pred_detailed_results
                        }
                        sample_predictions.append(pred_info)
                except Exception as e:
                    print(f"Error processing prediction {getattr(pred, 'id', 'unknown')}: {e}")
                    continue
            
            # 排序预测结果（最新的在前）
            sample_predictions.sort(key=lambda x: x['created_time'], reverse=True)
            sample_info['predictions'] = sample_predictions
            
            # 更新预测摘要信息 - 优先使用JSON文件中的数据
            if sample_info['json_predictions']:
                latest_json_pred = sample_info['json_predictions'][0]['data']
                rf_model = latest_json_pred.get('rf_model', {})
                sample_info['prediction_summary'] = {
                    'latest_risk_level': latest_json_pred.get('risk_level', '未知'),
                    'latest_confidence': rf_model.get('confidence', 0) * 100,
                    'prediction_count': len(sample_info['json_predictions']),
                    'latest_prediction_time': latest_json_pred.get('prediction_time', ''),
                    'autism_probability': rf_model.get('autism_probability', 0) * 100,
                    'control_probability': rf_model.get('control_probability', 0) * 100,
                    'source': 'json_file'
                }
            elif sample_predictions:
                latest_pred = sample_predictions[0]
                sample_info['prediction_summary'] = {
                    'latest_risk_level': latest_pred['risk_level'],
                    'latest_confidence': latest_pred['confidence'] * 100,
                    'prediction_count': len(sample_predictions),
                    'latest_prediction_time': latest_pred['created_time'],
                    'source': 'database'
                }
            
            # 检查是否有16S分析报告
            if sample['is_completed']:
                report_html_path = os.path.join(sample['path'], 'report.html')
                if os.path.exists(report_html_path):
                    sample_info['has_16s_report'] = True
                    sample_info['report_path'] = f"/media/{sample_name}/report.html"
            
            samples_data.append(sample_info)
        
        # 计算统计数据
        total_samples = len(user_samples)
        completed_analyses = len([s for s in user_samples if s['is_completed']])
        
        # 统计预测数量 - 包括JSON文件中的预测
        total_predictions = len(prediction_results)
        json_predictions_count = sum(len(s['json_predictions']) for s in samples_data)
        total_predictions = max(total_predictions, json_predictions_count)
        
        # 计算风险分布统计 - 优先使用JSON文件数据
        risk_distribution = {'高风险': 0, '中风险': 0, '低风险': 0}
        for sample in samples_data:
            if sample['json_predictions']:
                latest_json = sample['json_predictions'][0]['data']
                risk_level = latest_json.get('risk_level', '低风险')
                if risk_level in risk_distribution:
                    risk_distribution[risk_level] += 1
            elif sample['predictions']:
                risk_level = sample['predictions'][0]['risk_level']
                if risk_level in risk_distribution:
                    risk_distribution[risk_level] += 1
        
        # 计算平均置信度 - 优先使用JSON文件数据
        confidence_values = []
        for sample in samples_data:
            if sample['json_predictions']:
                latest_json = sample['json_predictions'][0]['data']
                rf_model = latest_json.get('rf_model', {})
                confidence = rf_model.get('confidence', 0)
                confidence_values.append(confidence)
            elif sample['predictions']:
                confidence = sample['predictions'][0]['confidence']
                confidence_values.append(confidence)
        
        if confidence_values:
            average_confidence = (sum(confidence_values) / len(confidence_values)) * 100
        else:
            average_confidence = 0
        
        context = {
            'samples_data': samples_data,
            'total_samples': total_samples,
            'completed_analyses': completed_analyses,
            'total_predictions': total_predictions,
            'risk_distribution': risk_distribution,
            'average_confidence': average_confidence,
        }
        
        return render(request, 'bioanalysis/sample_results.html', context)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Major error in sample_results view: {e}")
        
        # 在出错时显示错误信息
        from django.contrib import messages
        messages.error(request, f'加载样本结果时出错: {str(e)}')
        return redirect('dashboard')

@login_required
def predict_diseases(request):
    """处理常见病预测请求"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': '仅支持POST请求'})

    sample_name = request.POST.get('sample_name')
    if not sample_name:
        return JsonResponse({'success': False, 'error': '未指定样本名称'})

    try:
        # 检查样本是否存在且已完成分析
        available_samples = get_user_samples_with_status(request.user)
        sample = next((s for s in available_samples if s['name'] == sample_name and s['is_completed']), None)
        
        if not sample:
            return JsonResponse({'success': False, 'error': '样本不存在或尚未完成分析'})

        # 构建数据目录路径
        data_dir = os.path.join('media', sample_name)
        
        # 检查必要的文件是否存在
        required_files = ['Biom/OTU_table.tsv']
        for file in required_files:
            if not os.path.exists(os.path.join(data_dir, file)):
                return JsonResponse({'success': False, 'error': f'缺少必要文件: {file}'})

        # 记录预测开始活动
        log_user_activity(
            user=request.user,
            action_type='disease_prediction_start',
            description=f'开始常见病预测分析: {sample_name}',
            request=request
        )

        # 调用预测函数
        predictions = predict_multiple_diseases(data_dir)
        
        if predictions is False:
            return JsonResponse({'success': False, 'error': '预测过程中出错，请检查数据文件'})
        
        if not predictions:
            return JsonResponse({'success': False, 'error': '未获得有效的预测结果'})

        # 确保predictions是字典格式，且包含疾病名称和概率
        if not isinstance(predictions, dict):
            return JsonResponse({'success': False, 'error': '预测结果格式错误'})

        # 验证预测结果的有效性
        valid_predictions = {}
        for disease_name, probability in predictions.items():
            try:
                prob_value = float(probability)
                if 0 <= prob_value <= 1:
                    valid_predictions[disease_name] = prob_value
                else:
                    print(f"Invalid probability for {disease_name}: {prob_value}")
            except (ValueError, TypeError):
                print(f"Invalid probability format for {disease_name}: {probability}")
                continue
        
        if not valid_predictions:
            return JsonResponse({'success': False, 'error': '未获得有效的预测概率'})

        # 保存预测结果
        result = DiseasePredictionResult.objects.create(
            user=request.user,
            sample_name=sample_name
        )
        result.set_predictions(valid_predictions)
        result.save()

        # 记录预测完成活动
        log_user_activity(
            user=request.user,
            action_type='disease_prediction_complete',
            description=f'完成常见病预测分析: {sample_name}, 预测了{len(valid_predictions)}种疾病',
            request=request
        )

        return JsonResponse({
            'success': True,
            'predictions': valid_predictions,
            'total_diseases': len(valid_predictions),
            'sample_name': sample_name
        })

    except Exception as e:
        # 记录错误
        error_msg = str(e)
        print(f"Disease prediction error: {error_msg}")
        log_user_activity(
            user=request.user,
            action_type='error',
            description=f'常见病预测失败: {sample_name} - {error_msg}',
            request=request
        )
        return JsonResponse({'success': False, 'error': f'预测过程中发生错误: {error_msg}'})
