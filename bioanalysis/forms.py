from django import forms

class FileUploadForm(forms.Form):
    # 使用 FileField，multiple 属性需要在模板中手动添加
    file = forms.FileField()
