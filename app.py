from flask import Flask, render_template, request
from src.pipeline.prediction_pipeliene import CustomData, PredictPipeline

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@ app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Location=request.form.get('Location'),
            MinTemp=float(request.form.get('MinTemp')),
            MaxTemp=float(request.form.get('MaxTemp')),
            Rainfall=float(request.form.get('Rainfall')),
            Evaporation=float(request.form.get('Evaporation')),
            Sunshine=float(request.form.get('Sunshine')),
            WindGustSpeed=float(request.form.get('WindGustSpeed')),
            WindGustDir=request.form.get('WindGustDir'),
            WindDir9am=request.form.get('WindDir9am'),
            Humidity9am=float(request.form.get('Humidity9am')),
            Humidity3pm=float(request.form.get('Humidity3pm')),
            Pressure9am=float(request.form.get('Pressure9am')),
            Pressure3pm=float(request.form.get('Pressure3pm')),
            Cloud9am=float(request.form.get('Cloud9am')),
            Cloud3pm=float(request.form.get('Cloud3pm')),
            Temp9am=float(request.form.get('Temp9am')),
            Temp3pm=float(request.form.get('Temp3pm')),
            WindSpeed9am=float(request.form.get('WindSpeed9am')),
            WindSpeed3pm=float(request.form.get('WindSpeed3pm')),
            WindDir3pm=request.form.get('WindDir3pm')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results)
    
if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)

