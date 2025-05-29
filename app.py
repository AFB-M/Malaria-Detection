from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load your model
model = torch.load('malaria_resnet18.pth', map_location=torch.device('cpu'))
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Map prediction to label
labels = ['Parasitized', 'Uninfected']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file selected.")
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', prediction="No file selected.")
    
    img = Image.open(file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = labels[predicted.item()]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)