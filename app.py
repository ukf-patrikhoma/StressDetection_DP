from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load your model and tokenizer from the saved directory
model = BertForSequenceClassification.from_pretrained('./model_save')
tokenizer = BertTokenizer.from_pretrained('./model_save')
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        result = "Stress" if prediction == 1 else "No Stress"
        return render_template('index.html', user_input=user_input, result=result)
    return render_template('index.html', user_input='', result='')

if __name__ == '__main__':
    app.run(debug=True)
