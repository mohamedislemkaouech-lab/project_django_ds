from django.db import models

class KnowledgeItem(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    source = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
