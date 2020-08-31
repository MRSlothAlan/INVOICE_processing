from flask import Flask, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello world"


@app.route("/print_filename", methods=['POST', 'PUT'])
def print_filename():
    file = request.files['file']
    filename = secure_filename(file.filename)
    return filename


if __name__ == "__main__":
    app.run(port=5000, debug=True)

