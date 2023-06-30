from sklearn import preprocessing
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    '<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Prediction")


with st.form("my_form"):
    Income = st.number_input(label='Income', step=0.001, format="%.6f")

    Kidhome = st.number_input(label='Kidhome', step=0.001, format="%.6f")

    Teenhome = st.number_input(label='Teenhome', step=0.001, format="%.6f")

    Recency = st.number_input(label='Recency', step=0.001, format="%.6f")

    MntWines = st.number_input(label='MntWines', step=0.001, format="%.6f")

    MntFruits = st.number_input(label='MntFruits', step=0.001, format="%.6f")

    MntMeatProducts = st.number_input(
        label='MntMeatProducts', step=0.001, format="%.6f")

    MntFishProducts = st.number_input(
        label='MntFishProducts', step=0.001, format="%.6f")

    MntSweetProducts = st.number_input(
        label='MntSweetProducts', step=0.001, format="%.6f")

    MntGoldProds = st.number_input(
        label='MntGoldProds', step=0.001, format="%.6f")

    NumDealsPurchases = st.number_input(
        label='NumDealsPurchases', step=0.001, format="%.6f")

    NumWebPurchases = st.number_input(
        label='NumWebPurchases', step=0.001, format="%.6f")

    NumCatalogPurchases = st.number_input(
        label='NumCatalogPurchases', step=0.001, format="%.6f")

    NumStorePurchases = st.number_input(
        label='NumStorePurchases', step=0.001, format="%.6f")

    NumWebVisitsMonth = st.number_input(
        label='NumWebVisitsMonth', step=0.001, format="%.6f")

    AcceptedCmp3 = st.number_input(
        label='AcceptedCmp3', step=0.001, format="%.6f")

    AcceptedCmp4 = st.number_input(
        label='AcceptedCmp4', step=0.001, format="%.6f")

    AcceptedCmp5 = st.number_input(
        label='AcceptedCmp5', step=0.001, format="%.6f")

    AcceptedCmp1 = st.number_input(
        label='AcceptedCmp1', step=0.001, format="%.6f")

    AcceptedCmp2 = st.number_input(
        label='AcceptedCmp2', step=0.001, format="%.6f")

    Complain = st.number_input(label='Complain', step=0.001, format="%.6f")

    Response = st.number_input(label='Response', step=0.001, format="%.6f")

    Relationship_status = st.radio('Relationship_status', [
                                   'Single', 'Patner'])

    Education_status = st.radio(
        'Education_status', ['Graduate', 'Post_Graduate', 'Undergraduate'])

    Age = st.number_input(label='Age', step=0.001, format="%.6f")

    Total_kids = st.number_input(label='Total_kids', step=0.001, format="%.6f")

    Family_members = st.number_input(
        label='Family_members', step=0.001, format="%.6f")

    Total_Mnt = st.number_input(label='Total_Mnt', step=0.001, format="%.6f")

    Total_Purchases = st.number_input(
        label='Total_Purchases', step=0.001, format="%.6f")

    data = [[Income, Kidhome, Teenhome, Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases,
             NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, Relationship_status, Education_status, Age, Total_kids, Family_members, Total_Mnt, Total_Purchases]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust = loaded_model.predict(data)[0]
    print('Data Belongs to Cluster', clust)

    cluster_df1 = df[df['Cluster'] == clust]
    plt.rcParams["figure.figsize"] = (20, 3)
    for c in cluster_df1.drop(['Cluster'], axis=1):
        fig, ax = plt.subplots()
        grid = sns.FacetGrid(cluster_df1, col='Cluster')
        grid = grid.map(plt.hist, c)
        plt.show()
        st.pyplot(figsize=(5, 5))
