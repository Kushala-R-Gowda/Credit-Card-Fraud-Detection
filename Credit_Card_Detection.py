# Importing Dependencies
import numpy as np
from tkinter import *
import pandas as pd #To use dataframes to get structured data for analysis
from sklearn.model_selection import train_test_split #to spli our data into training data and split data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #to check accuracy of the model
from sklearn.preprocessing import LabelEncoder # Converting non numerical data type to numerical data


root = Tk()
root.title("CREDIT CARD FRAUD DETECTOR")
root.configure(background="powderblue")


def Model(Dataset):
	# separating the data for analysis
	legit = Dataset[Dataset.isFraud == 0] # rows with legit transactions
	fraud = Dataset[Dataset.isFraud == 1] # rows with fraud transactions
	
	# Taking a sample of Legit data using Under-Sampling to get similar distribution of both legit and fraud 			transactions
	# Number of Fraudulent Transaction = 8213
	legit_sample = legit.sample(n=8213)# this will extract 8213 datpoints randomly
	
	# Concatenate legit_sample and fraud dataframes
	new_Dataset = pd.concat([legit_sample,fraud], axis = 0) # axis= 0 -> add datapoints rowwise
	
	# Splitting the data into features and targets(either 0 or 1)
	x = new_Dataset.drop(columns=['isFraud','nameOrig','nameDest'],axis=1) # drops the the column isFraud and adds 		other columns to x
	y = new_Dataset['isFraud']
	
	# Split the data into training data and testing data
	X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,stratify = y, random_state=2) # x has 			features and y has label, 0.2(20%) of the x data is stored in X_test and its correspoding label is stored 		in Y_test and 80% of the x data is stored in X_train and its crresponding label is stored in Y_train
	# stratify is used to maintain similar distribution of data, random_state to split in some random way

	return X_train,X_test,Y_train,Y_test
	
def SingleInput(X_test,X_train,Y_train):
	model = LogisticRegression()
	model.fit(X_train,Y_train)
	SingleLinePredict = X_test.tail(1)
	prediction = model.predict(SingleLinePredict)
	#finaloutput = "Fraud : "+prediction[0]
	Oneframe = Frame(root,bg='orange',bd=2,pady=1,padx=1,width=400,height=800)
	Oneframe.place(x=-10,y=0)
	lbl=Label(Oneframe	, text=SingleLinePredict, fg= 'black' ,bg="white",font=("Helvetica", 26),width=200)
	lbl.pack(side=TOP)
	lbl=Label(Oneframe	, text=prediction, fg='red',bg="white", font=("Helvetica", 26),width=200)
	lbl.pack(side=TOPS)
	button = Button(Oneframe,text = "Click and Quit",bg="white", command = Oneframe.destroy)
	button.pack()
	
def FileInput(X_test,X_train,Y_train):
	model = LogisticRegression()
	model.fit(X_train,Y_train)
	FilePredict = X_test.tail(10)
	prediction = model.predict(FilePredict)
	#finaloutput = FilePredict+"\n"+prediction
	Oneframe = Frame(root,bg='black',bd=20,pady=5,relief=RIDGE)
	Oneframe.pack(side=TOP)
	lbl1=Label(Oneframe	, text=FilePredict, fg='black', font=("Helvetica", 26),width=200)
	lbl1.pack(side=TOP)
	lbl=Label(Oneframe	, text=prediction, fg='red', font=("Helvetica", 26),width=200)
	lbl.pack(side=TOP)
	button = Button(Oneframe,text = "Click and Quit", command = Oneframe.destroy)
	button.pack()
	
def ProjectInfo():
	frame = Frame(root,bg='olivedrab',bd=5,pady=1,relief=RAISED,width = 1000, height = 1200)
	frame.pack(side=TOP)
	frame1 = Frame(frame,bg='olivedrab',bd=5,pady=1,relief=RAISED,width = 500, height = 1000)
	frame1.pack(side=LEFT)
	frame2 = Frame(frame,bg='olivedrab',bd=5,pady=1,relief=RAISED,width = 500, height = 1000)
	frame2.pack(side=LEFT)
	file_name = PhotoImage(file = "info.png")
	buttonInfo1 = Button(frame1 , image = file_name ,width = 500, height = 1000)
	buttonInfo2 = Button(frame2 , image = file_name ,width = 500, height = 1000)
	button = Button(frame,text = "Click and Quit", command = frame.destroy)
	button.pack(side=BOTTOM)
	buttonInfo1.pack()
	buttonInfo2.pack()
	
	
def TeamMembers():
	frame = Frame(root,bg='olivedrab',bd=5,pady=1,relief=RAISED)
	frame.pack(side=TOP)
	frame1 = Frame(frame,bg='olivedrab',bd=5,pady=1,relief=RAISED)
	frame1.pack(side=BOTTOM)
	frame2 = Frame(frame,bg='olivedrab',bd=5,pady=1,relief=RAISED)
	frame2.pack(side=BOTTOM)
	file_name = PhotoImage(file = "sit.png")
	buttonNitin = Button(frame1 , image = file_name ,width = 200, height = 200)
	buttonVarun = Button(frame1 , image = file_name ,width = 200, height = 200)
	buttonNavdeep = Button(frame1 , image = file_name ,width = 200, height = 200)
	buttonSayeed = Button(frame1 , image = file_name ,width = 200, height = 200)
	buttonRitvik = Button(frame2 , image = file_name ,width = 200, height = 200)
	buttonKushala = Button(frame2 , image = file_name ,width = 200, height = 200)
	buttonMeghana = Button(frame2 , image = file_name ,width = 200, height = 200)
	buttonNidhi = Button(frame2 , image = file_name ,width = 200, height = 200)
	button = Button(frame,text = "Click and Quit", command = frame.destroy)
	buttonNitin.pack(side=RIGHT)
	buttonVarun.pack(side=RIGHT)
	buttonNavdeep.pack(side=RIGHT)
	buttonSayeed.pack(side=RIGHT)
	buttonRitvik.pack(side=RIGHT)
	buttonKushala.pack(side=RIGHT)
	buttonMeghana.pack(side=RIGHT)
	buttonNidhi.pack(side=RIGHT)
	button.pack(side=BOTTOM)
	
#Loading Dataset to pandas dataframe
Dataset = pd.read_csv("PS_20174392719_1491204439457_log.csv")
# Converting non numerical data type to numerical data
le = LabelEncoder()
Dataset.type = le.fit_transform(Dataset.type)
X_train,X_test,Y_train,Y_tes = Model(Dataset)
	
# Model Training - Logistic Regression
#model = LogisticRegression()
#model.fit(X_train,Y_train)

# HEADING LABEL
frame = Frame(root,bg='black',bd=20,pady=5,relief=RIDGE)
frame.pack(side=TOP)

textlabel = Label(frame , font=('calibri',40,'bold'),text='CREDIT CARD FRAUD DETECTOR',bd=15,bg="ivory",width = 40,fg="grey",justify=CENTER)
textlabel.grid(row=0)

# MAP PICTURE
frame1 = Frame(root,bg='olivedrab',bd=5,pady=1,relief=RAISED)
frame1.pack(side=BOTTOM)

file_name = PhotoImage(file = "sit.png")

buttonTeamMembers = Button(frame1 , image = file_name ,width = 200, height = 200, command=TeamMembers)
buttonInfo = Button(frame1 , image = file_name ,width = 200, height = 200, command=ProjectInfo	)
buttonFile = Button(frame1 , image = file_name ,width = 200, height = 200, command=lambda:FileInput(X_test,X_train,Y_train)	)
buttonOneLine = Button(frame1 , image = file_name ,width = 200, height = 200, command=lambda:SingleInput(X_test,X_train,Y_train))

buttonOneLine.pack(side=RIGHT)
buttonFile.pack(side=RIGHT)
buttonInfo.pack(side=RIGHT)
buttonTeamMembers.pack(side=RIGHT)

mainloop()
