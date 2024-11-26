from flask import request, Flask, render_template 
import pandas as pd 
import numpy as np
import pickle

app = Flask(__name__) # name for the Flask app (refer to output)

model1 = pickle.load(open('model.pkl', 'rb')) #loading my trained model
#model1 = joblib.load('filename.pkl') 

def drop(test_df):
  test_df.drop(["make"],axis=1,inplace=True)
  test_df.drop(["model"],axis=1,inplace=True)
  return test_df

def handle_categorical(test_df):           #this  function will handle all categorical features
  fuel_type_val= 'fuel_type' + '_' + test_df['fuel_type'][0]
  if fuel_type_val in test_df.columns:
    test_df[fuel_type_val] = 1             #replace 0 with 1 where the condition satifies or category meet

  condition_val= 'condition' + '_' + test_df['condition'][0]
  if condition_val in test_df.columns:
    test_df[condition_val] = 1

  transmission_val= 'transmission' + '_' + test_df['transmission'][0]
  if transmission_val in test_df.columns:
    test_df[transmission_val] = 1

  segment_val= 'segment' + '_' + test_df['segment'][0]
  if segment_val in test_df.columns:
    test_df[segment_val] = 1

  color_val= 'color' + '_' + test_df['color'][0]
  if color_val in test_df.columns:
    test_df[color_val] = 1

  drive_unit_val= 'drive_unit' + '_' + test_df['drive_unit'][0]
  if drive_unit_val in test_df.columns:
    test_df[drive_unit_val] = 1
  test_df.drop(["condition","fuel_type","transmission","color","segment","drive_unit"],axis=1,inplace=True)
  
  return test_df


@app.route('/')
def home():
    print('Applied Machine Learning Course')
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    print('Applied Machine Learning Course')
    features = request.form
    print(features)
    make = features['brand']
    model = features['model']
    year = features['year']
    condition = features['Condition']
    mileage = features['mileage']
    fuel_type = features['Fuel']
    volume = features['volume']
    color = features['colour']
    transmission = features['transmission']
    drive_unit = features['drive']
    segment = features['segment']


    user_input = {'make':[make],'model':[model],'year':[year],'condition':[condition],'mileage(kilometers)':[mileage],'fuel_type':[fuel_type],'volume(cm3)':[volume],'color':[color],'transmission':[transmission],'drive_unit':[drive_unit],'segment':[segment]}
    test_df = pd.DataFrame(user_input)

    #creating new dataframe for categorical feature and assuming all of them as 0 and replace it with 1 when handle categorical data through function calling. 
    new_df = pd.DataFrame(np.zeros(shape=(1,27)).astype(int),columns=(['condition_with damage', 'condition_with mileage', 'fuel_type_petrol',
       'transmission_mechanics', 'color_blue', 'color_brown', 'color_burgundy',
       'color_gray', 'color_green', 'color_orange', 'color_other',
       'color_purple', 'color_red', 'color_silver', 'color_white',
       'color_yellow', 'segment_B', 'segment_C', 'segment_D', 'segment_E',
       'segment_F', 'segment_J', 'segment_M', 'segment_S',
       'drive_unit_front-wheel drive', 'drive_unit_part-time four-wheel drive',
       'drive_unit_rear drive']))

    print(new_df)
    #concat both data frames 
    test_df = pd.concat([test_df,new_df],axis=1) 

    #Comment below two lines of code because i taken a user input data for a single car price prediction. You may uncomment it when deal with more data.
    # test_df = pre_processing_usertestcase(test_df)            
    # test_df = remove_outlier(test_df)

    test_df = drop(test_df)                #to drop unnecessary column
    test_df = handle_categorical(test_df)  #for encoding of categorical variable

    #make prediction using final data


    print(test_df)


    prediction = model1.predict(test_df)

    output = float(np.round(prediction[0], 2))

    print(output)

    return render_template('result.html', prediction_text='Predicted price of car is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)