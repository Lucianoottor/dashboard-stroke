from django.db import models

class Patient(models.Model):
    gender = models.CharField(max_length=6)
    age = models.IntegerField()
    hypertension = models.IntegerField()
    heart_disease = models.IntegerField()
    ever_married = models.CharField(max_length=3)
    work_type = models.CharField(max_length=20)
    Residence_type = models.CharField(max_length=6)
    avg_glucose_level = models.DecimalField(decimal_places=2, max_digits=10)
    bmi = models.DecimalField(decimal_places=1, max_digits=3)
    smoking_status = models.CharField(max_length=100)
    stroke = models.IntegerField()

    def __str__(self):
        return f"Patient {self.id} - {self.gender}, Age {self.age}"