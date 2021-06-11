from datetime import datetime, timedelta

def get_age(year, month, day):
   return datetime.today() - datetime(year, month, day)

def get_bmi(height, weight, units='imperial'):
    bmi = weight/(height**2)
    if units == 'imperial':
        return 703*bmi
    elif units == 'metric':
        return bmi
