from flask import Flask, request, render_template
import joblib
import webbrowser
import time
import threading

from train import clean_text

app = Flask(__name__)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = clean_text(request.form['message'])
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]

        result = 'Spam' if prediction == 1 else 'Ham'

        return render_template('index.html', result=result, message=message)

    return render_template('index.html', result=None, message=None)


def open_browser():
    time.sleep(1)  # wait for server to start
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=True, use_reloader=False)
