import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.metrics import mean_squared_log_error

@st.cache
def read_data():
    train_df = pd.read_csv("./dataset/training_data_v2.csv")
    return train_df

@st.cache
def read_data_two():
    norm_test_data = pd.read_csv("./dataset/normalised_test_data.csv")
    test_data = pd.read_csv("./dataset/test_data_v2.csv")
    return norm_test_data, test_data

@st.cache
def read_data_one():    
    raw_data = pd.read_csv("./dataset/toy_raw_set.csv")
    training_Data = pd.read_csv("./dataset/toy_training_set.csv")
    return raw_data, training_Data

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


# Train model 7  : oscilate, but fast decrease , Based on network22 ,  dropout to 0.3
class Net7(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(2048,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.hidden3 = nn.Linear(512,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.output = nn.Linear(128, output_size)
        
        self.relu = torch.nn.ReLU()
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):  # Input is a 1D tensor
        x = self.dropout1(self.relu(self.bn1(self.hidden1(x))))
        x = self.dropout2(self.relu(self.bn2(self.hidden2(x))))
        x = self.dropout3(self.relu(self.bn3(self.hidden3(x))))
        x = self.output(self.relu(x))
        return x
    

def load_model():
    # load all tensors onto the cpu
    # model = torch.load("./saved_models/Bestmodel.pth") 
    device = torch.device('cpu')
    model = Net7(39, 1)
    model.load_state_dict(torch.load("./saved_models/Bestmodel.pth", map_location=device))
    criterion = RMSLELoss()
    model.eval()
    return model, criterion

def scaler_y(df):
    scaler_y = StandardScaler(copy=True)
    scaler_y.fit(df['retweets'].to_numpy().reshape(-1,1))
    # new_scaled_y = scaler_y.transform(test['retweets'].to_numpy().reshape(-1,1))
    return scaler_y

def barchart(train_len, validation_len, test_len):
    # display bar chart
    st.subheader("Raw Dataset Distribution")
    labels = ["Training Set","Validation Set","Test Set"]
    data_rows = [train_len, validation_len, test_len]
    bc=pd.DataFrame(data_rows,index=labels)
    st.bar_chart(bc)


def data_visualisation(raw_data, training_Data):
    st.title("Data Visualisations")
    st.markdown("Top 100 rows of __Raw Dataset__")
    raw_data
    st.markdown("Top 100 rows of __Training Data__ after __preprocessing__")
    training_Data


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("50.021 AI Project")
    st.sidebar.subheader("COVID-19 Retweet Prediction")
    train_len = 1300207 
    test_len = 286811 
    validation_len = 325052
    train_df = read_data()

    user_choice = st.sidebar.radio("Navigation",['Data Visualisation','Try the Model here'])
    if user_choice != "Try the Model here":
        raw_data, training_Data = read_data_one()
        data_visualisation(raw_data, training_Data)
        barchart(train_len,validation_len,test_len)

        st.title("Scatter Plots")
        figu = px.scatter(raw_data,x='retweets',y='friends',color='retweets',hover_data=['followers','friends','favorites'],title='retweets and friends')
        fig = px.scatter(raw_data,x='retweets',y='followers',color='retweets',hover_data=['followers','friends','favorites'],title='retweets and followers')
        figu2 = px.scatter(raw_data,x='retweets',y='favorites',color='retweets',hover_data=['followers','friends','favorites'],title='retweets and favorites')
        st.plotly_chart(figu)
        st.plotly_chart(fig)
        st.plotly_chart(figu2)
    else:
        st.title("Model Prediction")
        norm_test_data, test_data = read_data_two()
        # print(len(norm_test_data),len(test_data))
        y_scaler = scaler_y(train_df)
        st.subheader("Which data would you like to choose for prediction?")
        # slider for display of each data
        selected_data = st.slider("",value=test_len//2,min_value=0,max_value=test_len-1)
        st.subheader("You have selected row {} from test set.".format(selected_data))
        newdf = test_data.iloc[[selected_data]]
        newdf
        normdf=norm_test_data.iloc[[selected_data]]
        # normdf
        
        # load model
        trained_model, criterion_ = load_model()
        # predict retweets
        input_features = torch.tensor(normdf.drop(["retweets"],1).to_numpy())
        
        # st.write(trained_model)
        # print(len(input_features[0]))
        y_pred = trained_model(input_features.float())
        y_pred_val = y_scaler.inverse_transform(y_pred.detach().numpy())
        
        y_label=torch.tensor(normdf['retweets'].to_numpy())
        y_label_val = y_scaler.inverse_transform(y_label.detach().numpy())
        # print(y_pred_val) #>>> [[val]]
        st.write("Predicted label:",y_pred_val[0][0])
        # print(y_label_val)
        st.write("Ground truth label:",y_label_val[0])
        loss_test=criterion_(y_pred,y_label.float().reshape(-1,1))
        st.write("MSLE Loss:",round(loss_test.item(),3))
        
        
        st.title("__Model Information__")
        st.image("./saved_models/output.png")

        # training & accuracy curve 
if __name__ == "__main__":
    main()