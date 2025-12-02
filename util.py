import numpy as np
from datetime import datetime

# 대곡교 티센 평균 강수량 ( 6개 지점 )
def thissen_DG(obs) :
    # 측정소 : 한국학중앙연구원, 대장동, 구미초교, 대곡교, 성남북초교, 남한산초교
    # 대곡교 유역 티센계수 (순서대로) 
    # 'R_한국학중앙연구원', 'R_대장동', 'R_구미초교', 'R_대곡교', 'R_성남북초교', 'R_남한산초교',
    weights_DG = np.array([0.251, 0.074, 0.125, 0.161, 0.373, 0.016])
    return np.sum(obs*weights_DG, axis=1)

# 궁내교 티센 평균 강수량 ( 3개 지점 )
def thissen_GN(obs) :      
    # 측정소 :  한국학중앙연구원, 대장동, 구미초교
    # 궁내교 유역 티센계수 (순서대로)
    # 'R_한국학중앙연구원', 'R_대장동', 'R_구미초교'
    weights_GN = np.array([0.061, 0.488 , 0.451])
    return np.sum(obs*weights_GN, axis=1)
    
    
# datetime -> 시간 문자열 변환
def get_datetime_str(base_date, date_format = '%Y-%m-%d %H:%M'):
    if isinstance(base_date, datetime):
        base_time = base_date.strftime(date_format)
    else : # 만약에 문자열이 들어왔으면 그대로 리턴
        base_time = base_date
    return base_time

# 문자열 날짜를 연월일시분 튜플로 파싱
def parse_date(year, month, day, hour, minute ) :
    # ex) 2024-08-29 24:00 -> 2024, 08, 30, 00, 00
    if month in [1, 3, 5, 7, 8, 10, 12]:
        max_day = 31
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:  # February
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            max_day = 29  # Leap year
        else:
            max_day = 28  # Non-leap year
    # Check if hour is 24 and adjust the date if necessary
    if hour == 24:
        hour = 0  # Reset the hour to 0 (midnight)
        if day < max_day:
            day += 1
        else:
            day = 1
            if month < 12:
                month += 1
            else:
                month = 1
                year += 1

    # Create the datetime object
    datetime_obj = datetime(year, month, day, hour, minute)

    # Format the datetime as a string
    #datetime_str = datetime_obj.strftime("%Y-%m-%d %H:%M")
    datetime_str = get_datetime_str(datetime_obj)
    return datetime_str