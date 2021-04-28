# Generated by Django 3.1.7 on 2021-04-06 01:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True
    dependencies = [
        ('accounts', '0001_initial'),
    ]
    operations = [
        migrations.CreateModel(
            name='Final_quiz',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quiz', models.CharField(max_length=200)),
                ('time', models.DateTimeField(auto_now_add=True)),
                ('kid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='accounts.kid')),
            ],
        ),
        migrations.CreateModel(
            name='Final_quiz_content',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.CharField(max_length=200)),
                ('test_type', models.IntegerField()),
                ('correct', models.BooleanField()),
                ('final_quiz', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='quizs.final_quiz')),
            ],
        ),
        migrations.CreateModel(
            name='Daily_quiz',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.CharField(max_length=50)),
                ('test_type', models.CharField(max_length=100)),
                ('try_time', models.IntegerField(default=1)),
                ('pic_dir', models.CharField(max_length=200)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('kid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='accounts.kid')),
            ],
        ),
    ]
