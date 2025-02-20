from flask import Flask,render_template,request
import pickle

application=Flask(__name__)
app=application

ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":

        temperature=float(request.form.get('temperature'))
        rh=float(request.form.get('rh'))
        ws=float(request.form.get('ws'))
        rain=float(request.form.get('rain'))
        ffmc=float(request.form.get('ffmc'))
        dmc=float(request.form.get('dmc'))
        isi=float(request.form.get('isi'))
        classes=float(request.form.get('classes'))
        region=float(request.form.get('region'))

        new_data_scaled=standard_scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        results=ridge_model.predict(new_data_scaled)

        return render_template("home.html",results=results[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")