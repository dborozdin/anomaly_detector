import streamlit as st
import pandas as pd
import numpy as np
from numpy import percentile
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use({'figure.facecolor':'white'})
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
import pytz
import matplotlib.dates as mdates
from sqlalchemy import create_engine
from sqlalchemy import text as sql_text

import time
#st.set_page_config(layout="wide")

st.title("Анализ аномалий")

RAND = np.random.RandomState(42)
CONTAMINATION = 0.01
outliers_fraction= CONTAMINATION
IFOREST_ESTIMATORS=200
DESICION_FUNC='max'
TIME_COLUMN='point'
FRAME_FROM='2024-04-19 0:00:00'
FRAME_TO='2024-04-20 0:00:00'

date_begin=FRAME_FROM
date_end=FRAME_TO
time_begin=dt.time()
time_end=dt.time()
analisysStarted = st.button('Начать анализ')

#параметры моделирования
method= 'CBLOF'
models= ['CBLOF', 'HBOS', 'COPOD', 'IForest']
ensemble_func='аномалия отмечена всеми моделями' 
#отображать или нет параметры
show_params= st.checkbox('Показать параметры')

if show_params:
    date_begin= st.date_input("Начальная дата временного окна", dt.datetime(2024, 4, 19), min_value=dt.datetime(2024, 4, 15), max_value= dt.datetime(2024, 4, 22), key='d_begin')
    time_begin= st.time_input("Начальное время", value=dt.time(), key='time_begin')
    date_end= st.date_input("Конечная дата временного окна", dt.datetime(2024, 4, 20), min_value=dt.datetime(2024, 4, 15), max_value= dt.datetime(2024, 4, 22), key='d_end')
    time_end= st.time_input("Конечное время", value=dt.time(), key='time_end')
    date_begin = dt.datetime.combine(date_begin.dt.tz_localize("Europe/Moscow"), time_begin)
    date_end = dt.datetime.combine(date_end.dt.tz_localize("Europe/Moscow"), time_end)
    st.write('date_begin:', date_begin)
    st.write('date_end:', date_end)
    outliers_fraction = st.slider(
        "Доля аномальных значений",
        0.0, 0.5, CONTAMINATION)
    method = st.selectbox(
        'Использовать метод поиска аномалий',
        models +  ['Ансамбль моделей'] )
    if method=='Ансамбль моделей (простой)':
        ensemble_func = st.selectbox(
        'Отметить значение как аномальное, если',
         ['аномалия отмечена всеми моделями', 'аномалия отмечена большей частью моделей', 'аномалия отмечена любой из моделей'])
         
def sql(query):
    full_db_name='sqlite:///my_lite_store_cut3.db'
    disk_engine_read = create_engine(full_db_name)
    df_sql = pd.read_sql_query(con=disk_engine_read.connect(), 
                                  sql=sql_text(query),
                                  parse_dates=TIME_COLUMN)#
    df_sql['point']= df_sql['point'].dt.tz_localize("Europe/Moscow").dt.tz_convert("Europe/Moscow")
    df_sql= df_sql.set_index('point')
    return df_sql
    
            
def get_anomaly_decision(scores, desicion_func='all'):
    cnt = 0 #количество предсказаний о наличии аномалий
    cnt_all = len(models)
    for col in models:
        if(scores[col]==1):
            cnt=cnt+1
    anomaly_decision = 0 #по умолчанию аномалии нет
    if desicion_func=='all': #все модели указывают на аномалию
        if cnt==cnt_all: 
            anomaly_decision= 1
    elif desicion_func=='max': #большинство (более половины) моделей указывают на аномалию
        if cnt/cnt_all>0.5: 
            anomaly_decision= 1
    else: #desicion_func=='any' #хотя бы одна модель указывает на аномалию
       if cnt>0: 
            anomaly_decision= 1     
    return anomaly_decision
    
my_tz = pytz.timezone('Europe/Moscow')
formatter = mdates.DateFormatter('%d.%m %H:%M', tz=my_tz)
def show_plot(df, anomaly_column):
    fig, axs = plt.subplots(len(df.columns), 1, sharex=True, constrained_layout=True, figsize=(8,6))
    anomal_time=anomaly_column[anomaly_column==1].index
    anomal_values=df.loc[anomal_time,:]

    col=df.columns[0]
    axs.plot(df[col], color='gray',label='Normal')
    axs.scatter(anomal_time, anomal_values[col], color='red', label='Anomaly')
    plt.xticks(rotation=45)
    axs.xaxis.set_major_formatter(formatter)
    #axs.xaxis_date()
    axs.set_title(col)
    st.pyplot(fig, use_container_width = True)
    
def calc(features, detectors):
    X=features.values
    table=pd.DataFrame(columns=models,index=features.index)
    desicion_func= DESICION_FUNC
    if len(detectors)<2:
        desicion_func='any'
    for detector in detectors:
        col= detector[1]
        table[col] = detector[0].fit(X).predict(X)
    return table.apply(lambda x: get_anomaly_decision(x, desicion_func), axis=1)    
    
if analisysStarted:
    bar = st.progress(0)
    latest_iteration = st.empty()
    latest_iteration.text('Получение данных из БД 0%')
    response=sql('select point,\
        sum(total_call_time) / sum(call_count)\
        from metrics\
        where\
            language = "java" \
            and app_name = "[GMonit] Collector"\
            and scope = "" \
            and name = "HttpDispatcher"\
            group by point \
            order by point')
    bar.progress(0.2)    
    latest_iteration.text('Получение данных из БД 20%')    
    throughput=sql('select point,\
                sum(call_count)\
                from metrics\
                where\
                    language = "java"\
                    and app_name = "[GMonit] Collector" \
                    and scope = ""\
                    and name = "HttpDispatcher"\
                    group by point\
                    order by point')       
    bar.progress(0.4)    
    latest_iteration.text('Получение данных из БД 40%')       
    ardex=sql('select point,\
                ((sum(call_count) + sum(total_call_time)/2) / (sum(call_count) + sum(total_call_time) + sum(total_exclusive_time)))\
                    from metrics\
                    where\
                        language = "java" \
                        and app_name = "[GMonit] Collector" \
                        and scope = ""\
                        and name = "Apdex" \
                        group by point \
                        order by point')
    bar.progress(0.6)   
    latest_iteration.text('Получение данных из БД 60%')       
    error_web=sql('SELECT point,\
                SUM(call_count) AS errors_web\
                FROM metrics\
                where\
                language = "java" \
                and app_name ="[GMonit] Collector"\
                and scope = ""\
                and name="Errors/allWeb"\
                group by point\
                order by point')      
    bar.progress(0.8)  
    latest_iteration.text('Получение данных из БД 80%')  
    http_call_cnt=sql('SELECT point,\
                SUM(call_count) AS http_call_cnt\
                FROM metrics\
                where\
                language = "java" \
                and app_name ="[GMonit] Collector"\
                and scope = ""\
                and name="HttpDispatcher"\
                group by point\
                order by point')    
    bar.progress(1)     
    latest_iteration.text('Данные загружены из БД')  
    time.sleep(1)
    latest_iteration.text(f'')
    features=response.join([throughput,ardex,error_web,http_call_cnt])
    features['errors_web']= features['errors_web'].fillna(0)
    features['errors']= features['errors_web']/features['http_call_cnt']
    features= features.drop(['errors_web', 'http_call_cnt'], axis=1)
    columns=['response','throughput','ardex', 'errors']
    features.columns=columns
    features.head()
         
    method_title= method
    if method=='Ансамбль моделей':
        method_title+= ' ('
        for model in models:
            method_title+= model
            method_title+= ', '
        method_title = method_title.rstrip(', ')
        method_title+= ')'
            
    #st.header(method_title)
    detectors=[]
    if method=='Ансамбль моделей':
        detectors_list=[]
        detectors_list.append(HBOS(contamination=outliers_fraction))
        detectors_list.append(CBLOF(contamination=outliers_fraction, random_state=RAND))
        detectors_list.append(COPOD(contamination=outliers_fraction))
        detectors_list.append(IForest(contamination=outliers_fraction, random_state=RAND, n_estimators=IFOREST_ESTIMATORS))
        for detector, model in zip(detectors_list, models):
            detectors.append((detector, model)) 
    elif method=='HBOS':
        detectors.append((HBOS(contamination=outliers_fraction), 'HBOS'))
    elif method=='COPOD':
        detectors.append((COPOD(contamination=outliers_fraction), 'COPOD'))
    elif method=='CBLOF':
        detectors.append((CBLOF(contamination=outliers_fraction, random_state=RAND), 'CBLOF'))
    else:# method=='IForest':
        detectors.append((IForest(contamination=outliers_fraction, random_state=RAND), 'IForest'))
    latest_iteration = st.empty()
    latest_iteration.text(f'Поиск аномалий и вывод диаграмм')
    bar.progress(1)
    i=0
    step= 1/len(columns)
    cnt=1
        
    for col in columns:
        df_feature= pd.DataFrame(features[col], columns=[col], index= features.index)
        anomalies=calc(df_feature, detectors)[date_begin:date_end]
        show_plot(df_feature[date_begin:date_end], anomalies)
    
        latest_iteration.text(f'Поиск аномалий и вывод диаграмм для {cnt} из {len(columns)}')
        bar.progress(i+step)
        i=i+step
        cnt=cnt+1
    latest_iteration.text(f'Готово!')
    time.sleep(3)
    latest_iteration.text(f'')