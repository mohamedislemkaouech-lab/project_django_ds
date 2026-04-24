from django.contrib import admin
from apps.orchestration.models import KnowledgeItem

@admin.register(KnowledgeItem)
class KnowledgeItemAdmin(admin.ModelAdmin):
    list_display = ("title", "source", "created_at")
    search_fields = ("title", "content")
