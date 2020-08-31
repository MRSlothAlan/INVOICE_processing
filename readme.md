# Read me for the invoice parsing system
Please make sure that you have installed all required packages.
```pip install -r "requirements.txt"```

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
    
