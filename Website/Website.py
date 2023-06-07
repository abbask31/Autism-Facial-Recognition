from fileinput import filename
from flask import *
from flask_dropzone import Dropzone
import numpy as np
import os, os.path
from PIL import Image
from skimage import transform
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class RecognitionModel(nn.Module):
    def __init__(self):
        super(RecognitionModel, self).__init__()

        # Load VGG19 model
        self.base = models.vgg19(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad = False
        
        # Modify the VGG19 classifier
        self.base.classifier = nn.Identity()  # Remove the default classifier layers
        
        # Define the remaining layers
        self.layers = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Apply the VGG19 base layers
        x = self.base.features(x)
        x = self.base.avgpool(x)
        
        # Apply the remaining layers
        x = self.layers(x)
        
        return x


# Initialize the folder where 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLD = 'Uploads\\'
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')

app = Flask(__name__, static_folder=UPLOAD_FOLDER) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device = torch.device('cpu')


# Load in the model
state_dict = torch.load('model.pt', map_location=torch.device('cpu'))
model = RecognitionModel()
model.load_state_dict(state_dict)
model.eval()

# Transform object used to resize and prepare each image
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Start the home page and initialize the dropzone
dropzone = Dropzone(app)
filenme = "NULL"
@app.route('/')  
def main():
    return reset()

# Main home page
@app.route('/upload', methods = ['POST'])  
def upload():
    if request.method == 'POST':  
        f = request.files['file']
        if ".jp" in f.filename:
            # Save the image for the website
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg"))

            global filenme 
            filenme = f.filename
        else:
            render_template("index.html", error = "File Is Not a JPG")
    return redirect(url_for('success'))

@app.route('/success', methods = ['POST'])  
def success():    
    # Open and transform the asved image
    try: image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg"))
    except:
        return render_template("index.html", error = "Invalid File Selected")
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    probability = None
    
    # Pass the image through the model
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
        class_index = predicted_class.item()  # Replace with the desired class index
        probability = probabilities[0, class_index]
    
    # Return the predicted class
    print(probability)
    diagnosis = "Class Prediction: Not Autistic"
    if predicted_class.item() == 0:
        diagnosis = "Class Prediction: Autistic"
    return render_template("upload.html", image = filenme, output = diagnosis)

# Resets the home page
@app.route('/reset', methods = ['POST'])  
def reset():
    global filenme 
    filenme = "NULL"
    #Try to delete the saved image
    try: os.remove(os.path.join(UPLOAD_FOLDER, "image.jpg"))
    except: pass
    return render_template("index.html")

# Apply the transform to the image
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

if __name__ == '__main__':  
    app.run(debug=True)