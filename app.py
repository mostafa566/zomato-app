import streamlit as st
import pickle
import numpy as np
import sklearn
import utilis

model=pickle.load(open('D:/training/first_project/pkl/model.pkl','rb'))
ohe=pickle.load(open('D:/training/first_project/pkl/One_Hot_Encoding.pkl','rb'))
ohe1=pickle.load(open('D:/training/first_project/pkl/One_Hot_Encoding1.pkl','rb'))
ohe2=pickle.load(open('D:/training/first_project/pkl/One_Hot_Encoding2.pkl','rb'))
le=pickle.load(open('D:/training/first_project/pkl/label_encoding.pkl','rb'))
std_scaler=pickle.load(open('D:/training/first_project/pkl/scaling.pkl','rb'))

location_list=['Banashankari',
 'Basavanagudi',
 'Mysore Road',
 'Jayanagar',
 'Kumaraswamy Layout',
 'Rajarajeshwari Nagar',
 'Vijay Nagar',
 'Uttarahalli',
 'JP Nagar',
 'South Bangalore',
 'City Market',
 'Nagarbhavi',
 'Bannerghatta Road',
 'BTM',
 'Kanakapura Road',
 'Bommanahalli',
 'CV Raman Nagar',
 'Electronic City',
 'HSR',
 'Marathahalli',
 'Sarjapur Road',
 'Wilson Garden',
 'Shanti Nagar',
 'Koramangala 5th Block',
 'Koramangala 8th Block',
 'Richmond Road',
 'Koramangala 7th Block',
 'Jalahalli',
 'Koramangala 4th Block',
 'Bellandur',
 'Whitefield',
 'East Bangalore',
 'Old Airport Road',
 'Indiranagar',
 'Koramangala 1st Block',
 'Frazer Town',
 'RT Nagar',
 'MG Road',
 'Brigade Road',
 'Lavelle Road',
 'Church Street',
 'Ulsoor',
 'Residency Road',
 'Shivajinagar',
 'Infantry Road',
 'St. Marks Road',
 'Cunningham Road',
 'Race Course Road',
 'Commercial Street',
 'Vasanth Nagar',
 'HBR Layout',
 'Domlur',
 'Ejipura',
 'Jeevan Bhima Nagar',
 'Old Madras Road',
 'Malleshwaram',
 'Seshadripuram',
 'Kammanahalli',
 'Koramangala 6th Block',
 'Majestic',
 'Langford Town',
 'Central Bangalore',
 'Sanjay Nagar',
 'Brookefield',
 'ITPL Main Road, Whitefield',
 'Varthur Main Road, Whitefield',
 'KR Puram',
 'Koramangala 2nd Block',
 'Koramangala 3rd Block',
 'Koramangala',
 'Hosur Road',
 'Rajajinagar',
 'Banaswadi',
 'North Bangalore',
 'Nagawara',
 'Hennur',
 'Kalyan Nagar',
 'New BEL Road',
 'Jakkur',
 'Rammurthy Nagar',
 'Thippasandra',
 'Kaggadasapura',
 'Hebbal',
 'Kengeri',
 'Sankey Road',
 'Sadashiv Nagar',
 'Basaveshwara Nagar',
 'Yeshwantpur',
 'West Bangalore',
 'Magadi Road',
 'Yelahanka',
 'Sahakara Nagar',
 'Peenya']

listedtype=['Buffet',
 'Cafes',
 'Delivery',
 'Desserts',
 'Dine-out',
 'Drinks & nightlife',
 'Pubs and bars']

listedcity=['Banashankari',
 'Bannerghatta Road',
 'Basavanagudi',
 'Bellandur',
 'Brigade Road',
 'Brookefield',
 'BTM',
 'Church Street',
 'Electronic City',
 'Frazer Town',
 'HSR',
 'Indiranagar',
 'Jayanagar',
 'JP Nagar',
 'Kalyan Nagar',
 'Kammanahalli',
 'Koramangala 4th Block',
 'Koramangala 5th Block',
 'Koramangala 6th Block',
 'Koramangala 7th Block',
 'Lavelle Road',
 'Malleshwaram',
 'Marathahalli',
 'MG Road',
 'New BEL Road',
 'Old Airport Road',
 'Rajajinagar',
 'Residency Road',
 'Sarjapur Road',
 'Whitefield']

def main():
    st.title("restaurant prediction success")
    
    # input variable
    online_order=st.selectbox("online order",["Yes","No"])
    book_table=st.selectbox("Book table",["Yes","No"])
    location=st.selectbox("location",location_list)
    rest_type=st.text_input("Rest type","Casual Dining")
    cuisines=st.slider("pick a number of cuisines",1,8)
    cost=st.number_input("approx cost of two people")
    typee=st.selectbox("select the Type of Restaurant",listedtype)
    city=st.selectbox("select the City of Restaurant",listedcity)
    
    #prepare data that is taken to model
    
    online_order=utilis.binary(online_order)
    book_table=utilis.binary(book_table)
    location=list(ohe.transform(np.array([[location]]))[0])
    rest_type=le.transform(np.array([[rest_type]]))
    cost=std_scaler.transform(np.array([[cost]]))
    typee=list(ohe1.transform(np.array([[typee]]))[0])
    city=list(ohe2.transform(np.array([[city]]))[0])
    
    if st.button("Predict"):
            # Perform prediction using the model and input variables
        input_data = [online_order]+[book_table]+ list(rest_type)+[cuisines]+list(cost[0])+typee+city+location
        prediction = model.predict(np.array([input_data]))[0]
        
        # Display the predicted result
        st.write("Prediction:", prediction)
        if prediction == 1 :
            st.success("Restaurant will be success")
        else:
            st.error("Restaurant will be failure")
    



    
if __name__=="__main__":
    main()
    