from flask import Flask, request, render_template, url_for
from langdetect import detect
from pathlib import Path
import spacy

# Load the urdu model
nlp_ur = spacy.load('../Custom Ner/for Urdu/output/model-best')
# Load the english model
nlp_en = spacy.load('../Custom Ner/for English/output/model-best')

def start_predict(doc):
    detector = detect(doc)
    if detector == "en":
        doc_en = nlp_en(doc)
        return doc_en
    else:
        doc_ur = nlp_ur(doc)
        return doc_ur



app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sent = request.form['sentences']
        doc_predict = start_predict(sent)
        ner_predicted = spacy.displacy.render(doc_predict, style='ent', page=True)
        output_path = Path("templates/results.html")
        output_path.open("w", encoding="utf-8").write(ner_predicted)
    return render_template('results.html')

if __name__ == "__main__":
    app.run()