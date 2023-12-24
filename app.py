import streamlit as st
import joblib


def main():
    st.title("Bank Term Deposit Prediction")
    
    input_data = {
            'age': st.number_input("Enter age:", min_value=0, max_value=100, value=18),
            'duration': st.number_input("Enter call duration in seconds:", min_value=0, max_value=4000, value=0, key='duration'),
            'campaign': st.number_input("Enter number of contacts performed during this campaign for this client:", min_value=0, max_value=50, value=0, key='campaign'),
            'previous': st.number_input("Enter number of contacts performed before this campaign for this client:", min_value=0, max_value=10, value=0, key='previous'),
            'emp.var.rate': st.number_input("Enter employment variation rate:", min_value=-5.0, max_value=5.0, value=0.0, step=0.01, key='emp.var.rate'),
            'cons.price.idx': st.number_input("Enter consumer price index:", min_value=0.0, max_value=105.0, value=0.0, step=0.1, key='cons.price.idx'),
            'cons.conf.idx': st.number_input("Enter consumer confidence index:", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key='cons.conf.idx'),    
            'euribor3m': st.number_input("Enter euribor 3 month rate:", min_value=0.0, max_value=7.0, value=0.0, step=0.001, key='euribor3m'),
            'nr.employed': st.number_input("Enter number of employees:", min_value=0, max_value=10000, value=0, step=10, key='nr.employed'),
            'job':  st.selectbox("Enter job:", ['blue-collar','services','admin.', 'entrepreneur', 'self-employed','technician','management','student','retired','housemaid','unemployed']),        
            'marital': st.selectbox("Enter marital status:", ['single', 'married', 'divorced']),
            'education': st.selectbox("Enter education level:", ['basic.4y','basic.6y','basic.9y', 'high.school', 'university.degree','professional.course','illiterate']),
            'default': st.selectbox("Has credit in default:", ['no', 'yes']),
            'housing': st.selectbox("Has a housing loan:", ['no', 'yes']),
            'loan': st.selectbox("Has a personal loan:", ['no', 'yes']),
            'contact': st.selectbox("Preferred contact type:", ['cellular', 'telephone']),
            'month': st.selectbox("Last contact month:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']),
            'day_of_week': st.selectbox("Last contact day of the week:", ['mon', 'tue', 'wed', 'thu', 'fri']),
    }

    if st.button("Operate"):
        prediction = perform_classification(input_data)
        st.subheader("Prediction:")
        st.write(prediction)


def perform_classification(input_data):
    model = joblib.load("my_bank_model.pkl")
    X = prepare_input_data(input_data)
    prediction = model.predict(X)
    label_map = {0: ("Not Suitable for term deposit", "rgba(255, 0, 0, 0.5)"), 1: ("Suitable for term deposit", "rgba(0, 255, 0, 0.5)")}
    prediction_label, prediction_color = label_map[prediction[0]]
    st.markdown(f'<div style="background-color: {prediction_color}; padding: 20px;">{prediction_label}</div>',
                unsafe_allow_html=True)
    return prediction_label

def prepare_input_data(input_data):
    import pandas as pd
    X = pd.DataFrame([input_data])
    cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in cols:
        X[col] = le.fit_transform(X[col])

    return X


if __name__ == "__main__":
    main()