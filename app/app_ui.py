import streamlit as st 
import requests
import pandas as pd

st.title("Credit Line Recommendation")

#Input Field
cust_id = st.number_input("Enter Customer ID", min_value = 1, step = 1)




#Button to trigger API call 
if st.button("Get Prediction"):
    try: 
        response = requests.get(f"http://127.0.0.1:8000/predict/?cust_id={cust_id}")
        
        if response.status_code == 200:
            data = response.json()
            if data:
                st.success("Prediction Results:")
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.warning(f"No prediction found for Customer ID {cust_id}")
        else:
            st.error(f"Error: {response.status_code} - {response.json().get('detail')}")
        
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        

        