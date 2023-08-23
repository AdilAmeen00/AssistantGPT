from flask import Flask, render_template, request, jsonify, Response
import privateGPT
import traceback
import io
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Redirect standard output to a StringIO buffer
        buffer = io.StringIO()
        sys.stdout = buffer

        message = request.form['message']
        response = privateGPT.run_model(message)
        
        # Reset standard output to its original value and get the content of the buffer
        sys.stdout = sys.__stdout__
        terminal_output = buffer.getvalue()
        buffer.close()

        # Return the response along with the terminal output
        return Response(f"{response}\n\nTerminal Output:\n{terminal_output}", content_type='text/plain; charset=utf-8')
    except Exception as e:
        print("Exception occurred:", str(e))
        exc_traceback = traceback.format_exc()
        return Response(f"Error: {str(e)}\n\nTraceback:\n{exc_traceback}", status=500, content_type='text/plain; charset=utf-8')


if __name__ == '__main__':
    app.run(debug=True)

