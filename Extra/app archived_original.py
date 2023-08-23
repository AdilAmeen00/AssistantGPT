from flask import Flask, render_template, request, jsonify
import privateGPT
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.form['message']
        response = privateGPT.run_model(message)
        return jsonify(response=response)
    except Exception as e:
        print("Exception occurred:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
