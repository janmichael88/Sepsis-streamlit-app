import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#CACHE FUNCTIONS
@st.cache
def dummy(nrows):
	print('')


#UN CAHCED
def pre_process(frame,method='linear'):
    frame = frame.interpolate(method=method).fillna(0)
    #assert nans
    return(frame)

def plot_sepsis_covariates(col_name,data,col1,col2):
    #create the above plot a colname
    fig,ax = plt.subplots(figsize=(20, 5))
    # make a plot
    ax.plot(data['Delta_t_hours'], data[col_name], color=col1, marker="o")
    # set x-axis label
    ax.set_xlabel("Hours",fontsize=14)
    # set y-axis label
    ax.set_ylabel(col_name,color="red",fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(data['Delta_t_hours'], data['SepsisLabel'],color=col2,marker="o")
    ax2.set_ylabel("Sepsis Onset",color="blue",fontsize=14)
    plt.title(col_name)
    st.pyplot(plt)


def plotly_two(col_name,data):
	# Create figure with secondary y-axis
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	# Add traces
	fig.add_trace(
	    go.Scatter(x=data['Delta_t_hours'], y=data[col_name], name=col_name),
	    secondary_y=False,
	)

	fig.add_trace(
	    go.Scatter(x=data['Delta_t_hours'], y=data['SepsisLabel'], name="Sepsis Onset"),
	    secondary_y=True,
	)

	# Add figure title
	fig.update_layout(
	    title_text="Sepsis Onset vs "+col_name
	)

	# Set x-axis title
	fig.update_xaxes(title_text="Hours in ICU")

	# Set y-axes titles
	fig.update_yaxes(title_text="<b>"+col_name+"</b>", secondary_y=False)
	fig.update_yaxes(title_text="<b>Sepsis Onset</b>", secondary_y=True)
	st.plotly_chart(fig)




#main call
def main():
	st.title('ICU Patient Septic Shock Detection and Data Explorer')

	#loading image
	st.image('demo_pic1.png',use_column_width=True)

	#Despcription
	st.markdown('''
		### Description
		* Sepsis is 11th leading cause of death for ICU patients. Aprroximately 1.7 million people develop sepsis and 270,00 die annually.
		* If patients do develop sepsis, it usually happens in the within 7 to 10 hours after admission.
		* This application assigns the probability a patient will expereience sepsis within the next 12 hours given a 36 hour observation window.
		* Current algorithm is Transformer LSTM with Attention (paper can be found here: [Attention is All you Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
		* The LSTM was trained on over 350,000 patient histories from the [MIMIC IV respository] (https://mimic-iv.mit.edu/)
		* INPUT: 36 hourly obervation window with 41 different indicators (i.e (36,41) tensor). 
		Indicators include Heart rate(beats per minut), Body Temperature (Deg C), Systolic BP (mm HG), etc. 
		* OUTPUT: Probability of Sepsis at each Hour (36,1 tensor)
		'''
		)
	#Show Columns
	st.markdown('### Column Descriptions')
	frame = pd.read_csv('Sepsis_Column_Descriptions.csv',index_col=0)
	st.write(frame)


	#First Section
	st.header("LOAD IN PATIENT SENSOR DATA")
	st.markdown('''
		* Ensure Data Input format is correct! If attaching your own csv file, ensure file is in same.
		* Demo application data available here on my [Github Repo](https://raw.githubusercontent.com/janmichael88/Sepsis_Detection/master/DemoTestSepsis.csv)
		* Upload your own sensor data, or use the sample provided.
		''')
	try:
		user_input = st.text_input("Path to csv file")
		data = pd.read_csv(user_input,index_col=0,encoding='utf-8')
		st.write(data)
	except FileNotFoundError:
		pass

	#Preoprcess Section
	st.header("PREPROCESS SENSOR DATA")
	if st.checkbox('Process Data? This will remove Nans and interpolate linearly.'):
		st.write(pre_process(data))

	#Data Explorere Section
	st.header('PATIENT SENSOR DATA EXPLORER')
	st.markdown('''
		* Red line indicates when patient went into sepsis (0,1).
		* Blue line can be toggled to select whatever covariate desired.
		* X axis is hours since patient has been in the ICU.
		''')
	if st.checkbox('Examine Time Series and Sepsis Onset Together?'):
		option = st.selectbox('What covariates do you wish to compare?',
			tuple(data.columns))
		plotly_two(option,data)

	#checkbox to create predictino vector
	st.header("DETERMINE RISK OF SEPSIS FOR THIS PATIENT?")
	if st.checkbox('Obtain Sepsis Probability trajectory in the next 12 hours? Takes 30 seconds!'):
		from Sepsis_Model import InstantiateModel
		foo = InstantiateModel()
		foo.create_arch()
		foo.input_weights('LSTM_finaloutput_v4.h5')
		#pull Xs
		Xs = data.iloc[:,1:-1].values.reshape((1,36,41))
		preds = foo.get_predictions(Xs)
		preds_frame = pd.DataFrame()
		preds_frame['Hours'] = list(range(36))
		preds_frame['Probability_Sepsis'] = preds
		st.write(preds_frame)

		#add in preds to original as prediction
		data['Probability_Sepsis'] = preds
		plotly_two('Probability_Sepsis',data)
		

if __name__ == "__main__":
	main()
