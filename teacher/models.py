from django.db import models
from django.utils import timezone
import datetime

class User(models.Model):

    # Member variables for User
    user_id = models.IntegerField(default=-1)
    score = models.IntegerField(default=-1)
    is_finished = models.BooleanField(default=False)
    start_time = models.DateTimeField(default=datetime.datetime.now())
    end_time = models.DateTimeField(default=datetime.datetime.now())
    graph = models.CharField(default='BASE', max_length=50)
    ucode = models.CharField(default='', max_length=15)

    # Function to construct a User
    @classmethod
    def create(cls, user_id_):
        user = cls(user_id = user_id_)
        return user


class UserResponse(models.Model):

    # Member variables for UserResponse
    user_id = models.IntegerField(default=-1)
    is_correct = models.BooleanField(default=False)

    # Function to construct a UserResponse
    @classmethod
    def create(cls, user_id_, is_correct_):
        user_response = cls(user_id = user_id_, is_correct = is_correct_)
        return user_response