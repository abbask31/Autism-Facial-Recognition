from distutils.log import debug
from fileinput import filename
from flask import *
from flask_dropzone import Dropzone
import os, os.path

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'Uploads\\'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_FOLD)

app = Flask(__name__, static_folder=UPLOAD_FOLDER) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

dropzone = Dropzone(app)
filenme = "NULL"
@app.route('/')  
def main():
    return reset()

@app.route('/upload', methods = ['POST'])  
def upload():
    if request.method == 'POST':  
        f = request.files['file']
        if ".jpg" in f.filename:
            f.save(os.path.join(UPLOAD_FOLDER, "image.jpg"))
            global filenme 
            filenme = f.filename
        else:
            print("Not a jpg")
            filenme = "Not a jpg"

@app.route('/success', methods = ['POST'])  
def success():
    return render_template("upload.html", image = filenme)

@app.route('/reset', methods = ['POST'])  
def reset():
    global filenme 
    filenme = "NULL"
    try: os.remove(os.path.join(UPLOAD_FOLDER, "image.jpg"))
    except: pass
    return render_template("index.html")

if __name__ == '__main__':  
    app.run(debug=True)