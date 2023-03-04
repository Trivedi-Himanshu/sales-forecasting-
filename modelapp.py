# Deployement model 


import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open(r'E:\Himanshu\DS project (Cement Sales and Demand)\360\flrmodel.pkl', 'rb'))


# import pandas as pd
# import streamlit as st 
# import numpy as np
# from statsmodels.regression.linear_model import OLSResults
# model = OLSResults.load("model1.pickle")

import matplotlib.pyplot as plt

# def main():
    

#     st.title("Forecasting")
#     st.sidebar.title("Forecasting")

#     # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
#     html_temp = """
#     <div style="background-color:tomato;padding:10px">
#     <h2 style="color:white;text-align:center;">Forecasting </h2>
#     </div>
    
#     """
#     st.markdown(html_temp, unsafe_allow_html = True)
#     st.text("")
    

#     uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
#     if uploadedFile is not None :
#         try:

#             data=pd.read_csv(uploadedFile,  index_col=0)
#         except:
#                 try:
#                     data = pd.read_excel(uploadedFile,  index_col=0)
#                 except:      
#                     data = pd.DataFrame(uploadedFile)
                
#     else:
#         st.sidebar.warning("you need to upload a 'csv' or 'excel' file.")
    
    
#     html_temp = """
#     <div style="background-color:tomato;padding:10px">
#     <p style="color:white;text-align:center;">Add DataBase Credientials </p>
#     </div>
#     """
#     # st.sidebar.markdown(html_temp, unsafe_allow_html = True)
#     if st.button("Predict"):
        
        
        
#         ###############################################
#         st.subheader(":red[Forecast for Test data]", anchor=None)
         
#         makeprediction = pd.DataFrame(model.predict(start = data.index[0], end = data.index[-1]))
#         results = pd.concat([data,makeprediction], axis=1)
#         import seaborn as sns
#         cm = sns.light_palette("blue", as_cmap=True)
#         st.table(results.style.background_gradient(cmap=cm).set_precision(2))
        
        
        
#         st.text("")
#         st.subheader(":red[Forcasted values for the given input...]", anchor=None)
        
#         forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + 12))
#         st.table(forecast.style.background_gradient(cmap=cm).set_precision(2))
    























if project == 'PREDICTION':








    def main():
    

    st.title("Forecasting")
    st.sidebar.title("Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile,  index_col=0)
        except:
                try:
                    data = pd.read_excel(uploadedFile,  index_col=0)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("you need to upload a 'csv' or 'excel' file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    # st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    if st.button("Predict"):
        
        
        
        ###############################################
        st.subheader(":red[Forecast for Test data]", anchor=None)
         
        makeprediction = pd.DataFrame(model.predict(start = data.index[0], end = data.index[-1]))
        results = pd.concat([data,makeprediction], axis=1)
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(results.style.background_gradient(cmap=cm).set_precision(2))
        
        
        
        st.text("")
        st.subheader(":red[Forcasted values for the given input...]", anchor=None)
        
        forecast = pd.DataFrame(model.predict(start=data.index[-1] + 1, end=data.index[-1] + 12))
        st.table(forecast.style.background_gradient(cmap=cm).set_precision(2))











    #st.sidebar.image('https://www.rabkindermpath.com/blog/admin/uploads/2020/rabkindx3.jpg')
    st.header('Prediction for Agent Arrival ')
    t = st.selectbox('give the value', data['Agent ID'].unique())
    slot = st.selectbox('Select Booking Slot', ['06:00 to 21:00 (Home)' , '19:00 to 22:00 (working person)', '06:00 to 18:00 (Collect at work place)'])
    gender = st.radio('Select Gender', ['Female', 'Male'])
    storage = st.selectbox('Specimen Storage', ['Vacuum blood collection tube', 'Urine culture transport tube', 'Disposable plastic container'])
    distance = np.log(st.number_input('Distance Between Patient and Agent in Meters'))
    collection_time = st.number_input('Specimen collection Time in minutes')
    patient_from = st.number_input('PATIENT AVAILABLE FROM', min_value=1, value=20)
    if st.checkbox('Show Instruction 1'):
        st.text('In "PATIENT AVAILABLE FROM" input the time when patient is available for test\n'
                'Eg.: patient is available from 13(1PM) to 14(2PM)\n'
                'Note: value should be in 24-hour format')
    patient_to = st.number_input('PATIENT AVAILABLE TO', min_value=1, value=21)
    if st.checkbox('Show Instruction 2'):
        st.text('In "PATIENT AVAILABLE TO" input the time when patient is available upto for test\n'
                'Eg.: patient is available from 13(1PM) to 14(2PM)\n'
                'Note: value should be in 24-hour format')
    agent_before = st.number_input('PATIENT ARRIVED BEFORE', min_value=1, value=21)
    if st.checkbox('Show Instruction 3'):
        st.text('Eg.: agent will reach before 14(2PM)')

    if st.button('Predict Timing'):

        if slot == '06:00 to 18:00 (Collect at work place)':
            slot = 0
        elif slot == '06:00 to 21:00 (Home)':
            slot = 1
        elif slot == '19:00 to 22:00 (working person)':
            slot = 2

        if gender == 'Female':
            gender = 0
        elif gender == 'Male':
            gender = 1

        if storage == 'Disposable plastic container':
            storage = 0
        elif storage == 'Urine culture transport tube':
            storage = 1
        elif storage == 'Vacuum blood collection tube':
            storage = 2

        query = np.array([agent_id, slot, gender, storage, distance, collection_time, patient_from, patient_to, agent_before])
        query = query.reshape(1, 9)

        result = model.predict(query)

        if result == 24:
            st.success(f'Agent will reached within {24} minutes')
        elif result == 34:
            st.success(f'Agent will reached within {34} minutes')
        elif result == 39:
            st.success(f'Agent will reached within {39} minutes')
        elif result == 49:
            st.success(f'Agent will reached within {49} minutes')
        elif result == 54:
            st.success(f'Agent will reached within {54} minutes')
        else:
            st.success(f'Agent will reached within {64} minutes')
            st.write('Your Location is to far')
        ## gif
        
if project == 'input_perticular_variables':
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# loading the trained model
pickle_in = open('flrmodel.pkl', 'rb')
flrmodel = pickle.load(pickle_in)

@st.cache()
def prediction(month, year, order_quantity_milliontonne):
    # create a DataFrame with the input data
    data = {'t': 0, 'aug': 0, 'dec': 0, 'feb': 0, 'jan': 0, 'jul': 0, 'jun': 0, 'mar': 0, 'may': 0, 'nov': 0, 'oct': 0, 'sep': 0}
    data[month.lower()] = 1
    data['year'] = year
    data['order_quantity_milliontonne'] = order_quantity_milliontonne
    df = pd.DataFrame(data, index=[0])
    
    # make the prediction
    prediction = flrmodel.predict(df)
    
    return prediction[0]

def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:orange;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Cement Sales Prediction ML App</h1> 
    </div> 
    """
    
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # following lines create boxes in which user can enter data required to make prediction
    month = st.selectbox("Select Month", months)
    year = st.text_input("Enter the Year")
    City = st.selectbox("select the city:", options=["Ahmedabad", "Aurangabad", "Bangalore", "Bhopal", "Chennai", "Coimbature", "Cuttack", "Guntur", "Hyderabad", "Jaipur", "Kochi", "Ludhiana", "Mangalore", "Mumbai", "Panaji", "Panipat", "Pune", "Surat", "Thrissur", "Tirupati", "Vijayawada"])
    Region = st.selectbox("select the region:", options=["Central", "Eastern", "Northern", "Southern", "Western"])
    ProductType = st.selectbox("select the Product type:", options=["OPC 43", "OPC 53", "PPC", "PSC"])
    order_quantity_milliontonne = st.slider("how much order quantity (Million tones)?", min_value=0.00, max_value=10.00, value=0.00, step=0.01)
    result = ""
    
    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(month, year, order_quantity_milliontonne)
        st.success('Your sale_quantity_milliontonne is {}'.format(result))
    












if __name__=='__main__':
    main() 