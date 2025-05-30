from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

# --- MODEL SETUP ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
checkpoint = torch.load('malaria_resnet18.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

labels = ['Malaria parasite detected', 'No malaria parasite found']

@app.route('/')
def index():
    print("[DEBUG] Loading index page: prediction=None, error=None")
    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("[DEBUG] No image part in request.files")
        return render_template('index.html', prediction=None, error="No file selected.")
    
    file = request.files['image']
    if file.filename == '':
        print("[DEBUG] Empty filename in upload")
        return render_template('index.html', prediction=None, error="No file selected.")
    
    img = Image.open(file).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out, 1)
        result = labels[pred.item()]
    
    print(f"[DEBUG] Prediction result: {result}")
    return render_template('index.html', prediction=result, error=None)

if __name__ == '__main__':
    app.run(debug=True)
