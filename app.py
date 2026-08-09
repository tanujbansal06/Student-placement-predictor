from flask import Flask,render_template,request
import joblib
app = Flask(__name__)
model = joblib.load("notebook/placement_model.pkl")
scaler = joblib.load("notebook/scaler.pkl")
@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/predictor")
def predictor():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    internship = int(request.form["internship"])
    project = int(request.form["project"])
    workshop = int(request.form["workshop"])
    aptitude = float(request.form["aptitude"])
    softskill = float(request.form["softskill"])
    extra = int(request.form["extra"])
    training = int(request.form["training"])
    ssc = int(request.form["ssc"])
    hsc = int(request.form["hsc"])

    features = [[
        cgpa,internship,project,workshop,aptitude,softskill,extra,training,ssc,hsc
    ]]
    features = scaler.transform(features)


    prob = model.predict_proba(features)

    placement_probability = prob[0][1] * 100
    if placement_probability < 25:
        message = "Very Low ~ You need significant improvement in your preparation."
    elif placement_probability < 50:
        message = "Low ~ You need more improvement in your skills and preparation."
    elif placement_probability < 70:
        message = "Moderate ~ Keep improving your skills and preparation."
    elif placement_probability < 85:
        message = "Good ~ You have a good chance of getting placed."
    else:
        message = "Excellent — You have a very strong chance of getting placed."

    return render_template(
    "result.html",
    probability=placement_probability,
    message=message
    )

if __name__ == "__main__":
    app.run(debug = True)
    
