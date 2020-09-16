from flask import Flask, request
from werkzeug.utils import secure_filename
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.parse_invoice_image import parse_main
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
import cv2
import numpy

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello world"


# test with syntax:
# curl -X POST -F file=@"C:/dummy.txt"
# http://localhost:5000/process_filename
@app.route("/process_file", methods=['POST', 'PUT'])
def process_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    img = cv2.imdecode(numpy.fromstring(request.files['file'].read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    parse_main(img=img, image_name=filename)
    return "done"


if __name__ == "__main__":
    app.run(port=5000, debug=True)

