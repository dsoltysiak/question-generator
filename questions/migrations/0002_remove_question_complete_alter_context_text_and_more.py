# Generated by Django 4.1.3 on 2022-12-04 09:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("questions", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="question",
            name="complete",
        ),
        migrations.AlterField(
            model_name="context",
            name="text",
            field=models.TextField(max_length=500),
        ),
        migrations.AlterField(
            model_name="question",
            name="context",
            field=models.OneToOneField(
                on_delete=django.db.models.deletion.CASCADE, to="questions.context"
            ),
        ),
        migrations.AlterField(
            model_name="question",
            name="question",
            field=models.TextField(max_length=300),
        ),
    ]
