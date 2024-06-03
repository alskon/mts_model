import streamlit as st
import requests
import json
import matplotlib.pyplot as plt

st.title('Модель предсказания оттока клиентов')
if 'trigger_result' not in st.session_state:
    st.session_state['trigger_result'] = False

if 'csv_predict' not in st.session_state:
    st.session_state['csv_predict'] = None

if 'json_features' not in st.session_state:
    st.session_state['json_features'] = None

if 'files_name' not in st.session_state:
    st.session_state['files_name'] = None

if 'hist_data' not in st.session_state:
    st.session_state['hist_data'] = None

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None and st.button('Run model'):
    bytes_data = uploaded_file.read()

    with st.spinner('Wait...'):

        response = requests.post('http://backend:5000/upload/', files={'file': bytes_data})

        response = requests.get('http://backend:5000/download/')
        if response.status_code == 200:
            files = json.loads(response.json())['files']
            hist_data = json.loads(response.json())['hist_data']
            st.session_state['files_name'] = files
            st.session_state['hist_data'] = hist_data
          

            response_csv = requests.get(f'http://backend:5000/download_file/{files[0]}')
            st.session_state['csv_predict'] = response_csv.content
            response_json = requests.get(f'http://backend:5000/download_file/{files[1]}')
            st.session_state['json_features'] = response_json.content
            st.session_state['trigger_result'] = True

if st.session_state['trigger_result']:
    btn_1 = st.download_button(label=':orange[Download prediction (csv)]',
                             data=st.session_state['csv_predict'],
                                 file_name=st.session_state['files_name'][0],
                                 key=st.session_state['files_name'][0])
    btn_2 = st.download_button(label=f'Download feature importances (json)',
                                 data=st.session_state['json_features'],
                                 file_name=st.session_state['files_name'][1],
                                 key=st.session_state['files_name'][1])
    fig, ax = plt.subplots()
    ax.hist(st.session_state['hist_data'], bins=20)
    ax.set_title('График плотности распределения')
    ax.set_xlabel('predict')
    ax.set_ylabel('count')
    plot_name = 'plot_result.png'
    fig.savefig(plot_name)
    with open(plot_name, "rb") as img:
      btn_3 = st.download_button(
                  label="Download plot (png)",
                  data=img,
                  file_name=plot_name,
                  mime="image/png"
              )
    st.pyplot(fig)




