import streamlit as st
import pandas as pd

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
# AUC
from sklearn.metrics import roc_auc_score

# Load the data
bank_test = pd.read_csv('bank_test.csv')

# result 列のfailを0、successを1に変換
bank_test['result'] = bank_test['result'].map({'fail': 0, 'success': 1})

# streamlitでファイルをアップロード
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# アップロードしたファイルを読み込み
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, dtype=int, header=None)

    st.write('Accuracy：', accuracy_score(bank_test['result'], input_df[0])*100, '%')
    st.write('Recall：', recall_score(bank_test['result'], input_df[0])*100, '%')
    st.write('Precision：', precision_score(bank_test['result'], input_df[0])*100, '%')
    st.write('F1score：', f1_score(bank_test['result'], input_df[0])*100, '%')
    st.write('AUC：', roc_auc_score(bank_test['result'], input_df[0])*100, '%')