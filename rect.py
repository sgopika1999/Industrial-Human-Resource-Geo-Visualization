import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from scipy.sparse import hstack
from sklearn.cluster import KMeans
import streamlit as st
import json
from PIL import Image

file_paths = [r"D:\\industrial\\Data1.csv",r"D:\\industrial\\Data2.csv" , r"D:\\industrial\\Data3.csv",
              r"D:\\industrial\\Data4.csv", r"D:\\industrial\\Data5.csv", r"D:\\industrial\\Data6.csv",
              r"D:\\industrial\\Data7.csv", r"D:\\industrial\\Data8.csv", r"D:\\industrial\\Data9.csv",
              r"D:\\industrial\\Data10.csv", r"D:\\industrial\\Data11.csv", r"D:\\industrial\\Data12.csv",
              r"D:\\industrial\\Data13.csv", r"D:\\industrial\\Data14.csv", r"D:\\industrial\\Data15.csv",
              r"D:\\industrial\\Data16.csv", r"D:\\industrial\\Data17.csv", r"D:\\industrial\\Data18.csv",
               r"D:\\industrial\\Data19.csv", r"D:\\industrial\\Data20.csv", r"D:\\industrial\\Data21.csv",
              r"D:\\industrial\\Data22.csv", r"D:\\industrial\\Data23.csv"]
dataframes = [pd.read_csv(file, encoding='latin1') for file in file_paths]
merged_df = pd.concat(dataframes, ignore_index=True)

# remove unnec comma, semi colon,
merged_df['State Code'] = merged_df['State Code'].astype(str).str.replace('`', '')
merged_df['District Code'] = merged_df['District Code'].astype(str).str.replace('`', '')
merged_df['Division'] = merged_df['Division'].astype(str).str.replace('`', '')
merged_df['Group'] = merged_df['Group'].astype(str).str.replace('`', '')
merged_df['Class'] = merged_df['Class'].astype(str).str.replace('`', '')

merged_df.columns = merged_df.columns.str.replace('-', '').str.replace(' ', '_').str.replace('__', '_')
merged_df["Total_workers"]=(merged_df["Main_Workers_Total__Persons"] + merged_df["Marginal_Workers_Total__Persons"])
numerical_cols = merged_df.select_dtypes(include=['number']).columns.tolist()

# Step 1: Select numerical columns
numerical_cols = merged_df.select_dtypes(include=['number']).columns.tolist()
for col in numerical_cols:
    data = merged_df[col]
    percentile25 = np.nanpercentile(data, 25)
    percentile75 = np.nanpercentile(data, 75)
    iqr = percentile75 - percentile25

    # Step 4: Calculate lower and upper bounds for capping
    lower_limit = percentile25 - 1.5 * iqr
    upper_limit = percentile75 + 1.5 * iqr

    # Step 5: Cap values outside the lower and upper bounds
    merged_df[col] = np.where(merged_df[col] > upper_limit, upper_limit, merged_df[col])
    merged_df[col] = np.where(merged_df[col] < lower_limit, lower_limit, merged_df[col])

numerical_cols = [
       'Main_Workers_Total__Persons',
       'Main_Workers_Total_Males', 'Main_Workers_Total_Females',
       'Main_Workers_Rural__Persons', 'Main_Workers_Rural_Males',
       'Main_Workers_Rural_Females', 'Main_Workers_Urban__Persons',
       'Main_Workers_Urban_Males', 'Main_Workers_Urban_Females',
       'Marginal_Workers_Total__Persons', 'Marginal_Workers_Total_Males',
       'Marginal_Workers_Total_Females', 'Marginal_Workers_Rural__Persons',
       'Marginal_Workers_Rural_Males', 'Marginal_Workers_Rural_Females',
       'Marginal_Workers_Urban__Persons', 'Marginal_Workers_Urban_Males',
       'Marginal_Workers_Urban_Females',"Total_workers"
]

X_numerical = merged_df[numerical_cols]
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X_numerical)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text data
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to text data
merged_df["NIC_Name"]=merged_df["NIC_Name"].apply(preprocess_text)
# Vectorizing the NIC Name column
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(merged_df["NIC_Name"])


# Combining text and numerical data
X_combined = hstack([X_text, X_numerical])

# Determine the optimal number of clusters using the Elbow method
def plot_elbow_method(X, max_clusters=10):
    sse = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters+1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal Number of Clusters')
    

plot_elbow_method(X_combined)
num_cluster=5
kmeans = KMeans(n_clusters=num_cluster, random_state=42)
clusters = kmeans.fit_predict(X_combined)
merged_df['Cluster'] = clusters

st.set_page_config(page_title="INDUSTRIAL HUMAN RESOURCE GEO-VISUALIZATION",layout="wide")
st.header("INDUSTRIAL HUMAN RESOURCE GEO-VISUALIZATION")
tab1,tab2=st.tabs(["INTRODUCTION","DATA EXPLORATION"])
default_option="INTRODUCTION"
with tab1:
    col1,col2 = st.columns(2)
    with col1:
        st.write("")
        st.write("")
        st.image(Image.open("C:\\Users\\andro\\Downloads\\hr.jpg"), width=500)
        st.markdown("#### :red[**Technologies Used :**] Python, Pandas, Visualization, Streamlit, Scikit-learn, NLTK")
    with col2:
        st.write("#### :red[**Overview :**] Industrial Human Resource Geo-Visualization refers to the process of visually representing data related to the workforce within various industries across different geographical locations.  In this visualization aims to provide insights into the distribution, characteristics, and trends of the industrial workforce in a way that is easy to interpret and analyze")


with tab2:
   

    plt.figure(figsize=(20, 6))
    sns.countplot(x='State_Code', data=merged_df)
    plt.title('Count Plot of State_code')
    plt.xlabel('State_code')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.write('''
                Rajasthan: Holds the highest number of workers.
                Maharashtra: Ranks second in the number of workers.
                Tamil Nadu: Comes in third in the number of workers.''')

    col1,col2=st.columns(2)
    with col1:
        total_workers=merged_df[[
         'Main_Workers_Rural_Males',
        'Main_Workers_Rural_Females', 
        'Main_Workers_Urban_Males', 'Main_Workers_Urban_Females'
        ]].agg('mean').reset_index()
        total_workers.columns = ['Category', 'Count']
        fig = px.bar(total_workers, x='Category', y='Count', title='Main Workers Distribution')
        fig.update_layout(
            xaxis_title='Category',
            yaxis_title='Number of Workers',
            legend_title_text='Category',
            xaxis_tickangle=-45  
        )
        st.plotly_chart(fig)

    with col2:
        total_workers=merged_df[[ 
        'Marginal_Workers_Rural_Males', 'Marginal_Workers_Rural_Females',
            'Marginal_Workers_Urban_Males','Marginal_Workers_Urban_Females']].agg('mean').reset_index()
        total_workers.columns = ['Category', 'Count']
        fig = px.bar(total_workers, x='Category', y='Count', title='Marginal Workers Distribution')
        fig.update_layout(
            xaxis_title='Category',
            yaxis_title='Number of Workers',
            legend_title_text='Category',
            xaxis_tickangle=-45  
        )
        st.plotly_chart(fig)
    
    st.write('''1. There are more male workers than female workers in both rural and urban areas across main and marginal work categories.
                2. The number of female workers is higher in the marginal category than in the main category for both rural and urban areas.
                3. There are fewer main workers in rural areas compared to urban areas.
                4. There are more marginal workers in rural areas compared to urban areas.''')



    excluded_values = ["total", "blank","incomplete wrongly classifed"]
    merged_df["NIC_Name"] = merged_df["NIC_Name"].str.strip()
    filtered_df = merged_df[~merged_df["NIC_Name"].str.lower().isin(excluded_values)]
    filtered_df['Total_workers'] = filtered_df['Main_Workers_Total__Persons'] + filtered_df['Marginal_Workers_Total__Persons']
    NIC = filtered_df.groupby("NIC_Name")["Total_workers"].agg('mean').reset_index()
    NIC_final = NIC.sort_values(by='Total_workers', ascending=False)
    top_20 = NIC_final.head(20)
    fig_bar = px.bar(
    top_20,
    x="NIC_Name",
    y="Total_workers",
    title="Top 20 NIC by Total Workers",
    color_discrete_sequence=px.colors.sequential.Redor_r,
    width=600,
    height=500
)

    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar)

    st.write("From the above analysis I found that majorly working industries are Retail trade,public administration ,land transport and construction building")    
    
    col1,col2=st.columns(2)
    with col1:
        excluded_values = ["total", "blank", "incomplete wrongly classifed"]
        merged_df["NIC_Name"] = merged_df["NIC_Name"].str.strip()
        filtered_df1 = merged_df[~merged_df["NIC_Name"].str.lower().isin(excluded_values)]
        filtered_df1["Total_Female_Workers"] = filtered_df1['Main_Workers_Total_Females'] + filtered_df1["Marginal_Workers_Total_Females"]
        NIC=filtered_df1.groupby("NIC_Name")["Total_Female_Workers"].agg('mean').reset_index()
        NIC_final = NIC.sort_values(by='Total_Female_Workers', ascending=False)
        top_5_Female = NIC_final.head(5)
        
        fig_bar = px.bar(
            top_5_Female,
            x="NIC_Name",
            y="Total_Female_Workers",
            title="Top NIC by total Female workers",
            color_discrete_sequence=px.colors.sequential.Redor_r,
            width=600,
            height=500
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar)

        st.write("Female prefers to work in human health activity,in education and in service based sectors and these are the top 3 indutries.")

    with col2:
       
        merged_df["NIC_Name"] = merged_df["NIC_Name"].str.strip()
        filtered_df2 = merged_df[merged_df["NIC_Name"].str.lower() != "total"]
        filtered_df2["Total_Male_Workers"] = filtered_df2["Main_Workers_Total_Males"] + filtered_df2["Marginal_Workers_Total_Males"]
        NIC = filtered_df2.groupby("NIC_Name")["Total_Male_Workers"].agg('mean').reset_index()
        NIC_final = NIC.sort_values(by='Total_Male_Workers', ascending=False)
        top_5_Male = NIC_final.head(5)
        fig_bar = px.bar(
            top_5_Male,
            x="NIC_Name",
            y="Total_Male_Workers",
            title="Top NIC by total Male workers",
            color_discrete_sequence=px.colors.sequential.Redor_r,
            width=600,
            height=500
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar)
        st.write("Male prefers to work in land transport, public administration, retail trade and construction building.")
            
    st.title("Industry Names by Cluster")
    col1,col2,col3=st.columns(3)
    with col1:
        cluster_df = merged_df[merged_df['Cluster'] == 0][['NIC_Name']]
        st.subheader("Cluster 0")
        st.dataframe(cluster_df,height=200) 

    with col2:
        cluster_df = merged_df[merged_df['Cluster'] == 1][['NIC_Name']]
        st.subheader("Cluster 1")
        st.dataframe(cluster_df,height=200)

    with col3:
        cluster_df = merged_df[merged_df['Cluster'] == 2][['NIC_Name']]
        st.subheader("Cluster 2")
        st.dataframe(cluster_df,height=200)

    col1,col2=st.columns(2)
    with col1:
        cluster_df = merged_df[merged_df['Cluster'] == 3][['NIC_Name']]
        st.subheader("Cluster 3")
        st.dataframe(cluster_df,height=200)

    with col2:
        cluster_df = merged_df[merged_df['Cluster'] == 4][['NIC_Name']]
        st.subheader("Cluster 4")
        st.dataframe(cluster_df,height=200)

    col1,col2=st.columns(2)
    with col1:
        cluster = st.selectbox("Select the clusters", merged_df["Cluster"].unique())
        df1 = merged_df[merged_df["Cluster"] == cluster]

    with col2:
    
        nic = st.selectbox("Select the Industry name", df1["NIC_Name"].unique())
        df2 = df1[df1["NIC_Name"] == nic]

        country = df2.groupby('State_Code', as_index=False).agg({
            'Main_Workers_Total__Persons': 'mean',
            'Marginal_Workers_Total__Persons': 'mean',
            'India/States': "first",
        'Main_Workers_Total_Males':'mean', 
        'Main_Workers_Total_Females':'mean',
       'Main_Workers_Rural__Persons':'mean', 
       'Main_Workers_Rural_Males':'mean',
       'Main_Workers_Rural_Females':'mean', 
       'Main_Workers_Urban__Persons':'mean',
       'Main_Workers_Urban_Males':'mean', 
       'Main_Workers_Urban_Females':'mean', 
       'Marginal_Workers_Total_Males':'mean',
       'Marginal_Workers_Total_Females':'mean', 
       'Marginal_Workers_Rural__Persons':'mean',
       'Marginal_Workers_Rural_Males':'mean', 
       'Marginal_Workers_Rural_Females':'mean',
       'Marginal_Workers_Urban__Persons':'mean', 
       'Marginal_Workers_Urban_Males':'mean',
       'Marginal_Workers_Urban_Females':'mean'})

   
    with open("C:\\Users\\andro\\Downloads\\states_india.geojson", 'r') as file:
        geojson_data = json.load(file)
    
    st.write("GOEGRAPHIC VISUALIZATION")
    fig = px.choropleth(country,
                        geojson=geojson_data,
                        featureidkey='properties.state_code',  
                        locations='State_Code',
                        hover_data={'India/States':True,
                                    'Main_Workers_Total_Males':True,
                                    'Main_Workers_Total_Females':True},
                        color='Main_Workers_Total__Persons',
                        color_continuous_scale="Viridis",width=1000,height=400,
                        
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)

    
    fig = px.choropleth(country,
                        geojson=geojson_data,
                        featureidkey='properties.state_code',  
                        locations='State_Code',
                        hover_data={'India/States':True,
                                    'Marginal_Workers_Total_Males':True,
                                    'Marginal_Workers_Total_Females':True},
                        color='Marginal_Workers_Total__Persons',
                        color_continuous_scale="Viridis",width=1000,  
                        height=400, 
                        )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)

 

    method=st.radio('Select Worker Type', ['Main Workers', 'Marginal Workers'])
    if method=='Main Workers':
        df_melted = pd.melt(country, id_vars=["State_Code"], value_vars=['Main_Workers_Total_Males', 'Main_Workers_Total_Females'],
                    var_name='Worker_Type', value_name='Total_Workers')

        # Plot using seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 8))
        sns.barplot(x="State_Code", y="Total_Workers", hue="Worker_Type", data=df_melted, palette="viridis")

        # Add title and labels
        plt.title('Clustered Bar Chart')
        plt.xlabel('State Code')
        plt.ylabel('Total Workers')

        # Display the plot
        st.pyplot(plt)

        col1,col2=st.columns(2)
        with col1:
            st.write("MAIN URBAN WORKERS")

            df_melted = pd.melt(country, id_vars=["State_Code"], value_vars=['Main_Workers_Urban_Males', 'Main_Workers_Urban_Females'],
                        var_name='Worker_Type', value_name='Total_Workers')
            sns.set(style="whitegrid")
            plt.figure(figsize=(14, 8))
            sns.barplot(x="State_Code", y="Total_Workers", hue="Worker_Type", data=df_melted, palette="bright")
            plt.title('Clustered Bar Chart')
            plt.xlabel('State Code')
            plt.ylabel('Total Workers')
            st.pyplot(plt)

        with col2:
            st.write("MAIN RURAL WORKERS")
        
            df_melted = pd.melt(country, id_vars=["State_Code"], value_vars=['Main_Workers_Rural_Males',
                        'Main_Workers_Rural_Females'],var_name='Worker_Type', value_name='Total_Workers')
            sns.set(style="whitegrid")
            plt.figure(figsize=(14, 8))
            sns.barplot(x="State_Code", y="Total_Workers", hue="Worker_Type", data=df_melted, palette="bright")
            plt.title('Clustered Bar Chart')
            plt.xlabel('State Code')
            plt.ylabel('Total Workers')
            st.pyplot(plt)

    if method=='Marginal Workers':
        df_melted = pd.melt(country, id_vars=["State_Code"], value_vars=['Marginal_Workers_Total_Males',
                    'Marginal_Workers_Total_Females'],var_name='Worker_Type', value_name='Total_Workers')
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 8))
        sns.barplot(x="State_Code", y="Total_Workers", hue="Worker_Type", data=df_melted, palette="viridis")
        plt.title('Clustered Bar Chart')
        plt.xlabel('State Code')
        plt.ylabel('Total Workers')
        st.pyplot(plt)

        col1,col2=st.columns(2)
        with col1:
            st.write("MARGINAL URBAN WORKERS")

            df_melted = pd.melt(country, id_vars=["State_Code"], value_vars=['Marginal_Workers_Urban_Males',
                        'Marginal_Workers_Urban_Females'],var_name='Worker_Type', value_name='Total_Workers')
            sns.set(style="whitegrid")
            plt.figure(figsize=(14, 8))
            sns.barplot(x="State_Code", y="Total_Workers", hue="Worker_Type", data=df_melted, palette="bright")
            plt.title('Clustered Bar Chart')
            plt.xlabel('State Code')
            plt.ylabel('Total Workers')
            st.pyplot(plt)

        with col2:
            st.write("MARGINAL RURAL WORKERS")

            df_melted = pd.melt(country, id_vars=["State_Code"], value_vars=['Marginal_Workers_Rural_Males', 'Marginal_Workers_Rural_Females'],
                        var_name='Worker_Type', value_name='Total_Workers')
            sns.set(style="whitegrid")
            plt.figure(figsize=(14, 8))
            sns.barplot(x="State_Code", y="Total_Workers", hue="Worker_Type", data=df_melted, palette="bright")
            plt.title('Clustered Bar Chart')
            plt.xlabel('State Code')
            plt.ylabel('Total Workers')
            st.pyplot(plt)
         

    

   