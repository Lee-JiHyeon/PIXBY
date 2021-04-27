from django.contrib import admin
from .models import User, EnvSettings, Kid, StudyRecord
# Register your models here.


admin.site.register(User)
admin.site.register(EnvSettings)
admin.site.register(Kid)
admin.site.register(StudyRecord)