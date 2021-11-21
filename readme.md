# Read me for the invoice parsing system
This is a fun side-project fyr.

Please make sure that you have installed all required packages.
```pip install -r "requirements.txt"```

# About Tesseract-OCR, and other missing directory and data
The repo is for the presentation of the logic. 
Other directories, such as /Tesseract-OCR, and some testing code / dataset are ignored.
Datasets are removed due to privacy concerns.

The list of removed directories:

* INVOICE_PROCESSING/Tesseract-OCR/
* playground/
* related_research/
* INVOICE_PROCESSING/test_images/
* INVOICE_PROCESSING/output_json/
* INVOICE_PROCESSING/output_csv/
* INVOICE_PROCESSING/OpenCV_EAST/

# Demo:
### Input invoice
![Alt text](demo/original_invoice.png?raw=true "Original invoice")

### Output csv
Tables (with or without border / features) are stored in a CSV file.
![Alt text](demo/result_csv.png?raw=true "CSV")

### Output json
Other entries are exported to a json file
![Alt text](demo/result_json.png?raw=true "JSON")

### Work done:
#### Parse invoices in local directory
##### A script that parse invoice in local directory, perform whitening and alignment, parse invoices, return a json file
##### Instruction to run it:
1.  run the parse_invoice_dataset.py script
2.  set the paths in the script accordingly if path error exists
3.  the script scans all files stored in "test_images" folder
4.  Json files are returned and saved in "output_json" folder
#### Parse invoices by calling api
##### A minimal implementation of the api is done. 
##### Instructions to run it:
1.  run api_main.py
2.  send a POST request to the localhost / hosted environment. The default option is running at localhost, using port 5000. The command on windows cmd to make a POST request is:
    ```curl -X POST -F file=@"Path/to/image.jpg" http://localhost:5000/process_file```
3. a json file is returned and saved to the "output_json" folder
       
### Work in progress:
<li>A reasonable machine learning approach to detect entries and line items. Features must be selected carefully</li>
<li>pdf to image is needed. Will update this functionality very soon.</li>
    
