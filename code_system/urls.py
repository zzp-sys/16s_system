"""
URL configuration for code_system project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

def root_redirect(request):
    """根URL重定向 - 已登录用户到整合分析页面，未登录用户到登录页面"""
    if request.user.is_authenticated:
        return redirect('/integrated/')
    else:
        return redirect('/login/')

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', root_redirect, name='root'),
    path('', include('bioanalysis.urls')),
]

# 在开发环境中提供静态文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)