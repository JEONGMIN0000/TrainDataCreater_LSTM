import requests
import util
from typing import List
# 홍수통제소 에서 가져온 json을 필요한 데이터만 추출해서 list로 변환
def hrfcodict_list (datatype, code_area, start_date, end_date ) -> List[dict] :
    # call function
    # 강우 : collect.hrfcodict_list ('rf', 관측소코드, 검색시작일시, 검색종료일시)
    # 수위 : collect.hrfcodict_list ('wl', 관측소코드, 검색시작일시, 검색종료일시)
    # return : List [ {time, rf} ] || List [{time, wl, fw}]
    
    #홍수통제소에서 
    hrfco_key = '20206284-E899-4B70-98EF-B77872B9AD86' # 홍수통제소 api 키

    if datatype == 'rf' : # 강우 데이터
        url = f'https://api.hrfco.go.kr/{hrfco_key}/rainfall/list/10M/{code_area}/{start_date}/{end_date}.json'        
    elif datatype == 'wl': # 수위 데이터
        url = f'https://api.hrfco.go.kr/{hrfco_key}/waterlevel/list/10M/{code_area}/{start_date}/{end_date}.json'

    response = requests.get(url, params={}) 
    hrfco_list = response.json()['content']
    past_list = []  

    def splitDatetime (time) :
        return int(time[:4]), int(time[4:6]), int(time[6:8]), int(time[8:10]), int(time[10:12])

    if datatype == 'rf' : # 강우 데이터
        for item in hrfco_list :
            year, month, day, hour, minute = splitDatetime(item['ymdhm'])            
            
            inner_data = {
                'time' : util.parse_date(year, month, day, hour, minute),
                'rf' : item['rf']
            }
            past_list.append(inner_data)
    elif datatype == 'wl' : #수위
        for item in hrfco_list :
            year, month, day, hour, minute = splitDatetime(item['ymdhm'])     
            
            inner_data = {
                'time' : util.parse_date(year, month, day, hour, minute),
                'wl' : item['wl'],# 수위
                'fw' : item['fw'],# 유량
            }
            
            past_list.append(inner_data)
    return past_list 