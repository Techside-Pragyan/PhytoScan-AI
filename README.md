# 🌱 PhytoScan AI - Plant Disease Identifier

PhytoScan AI is a full-stack, end-to-end Deep Learning web application designed to identify plant diseases from leaf images and provide actionable treatment suggestions. 

It leverages a pre-trained **MobileNetV2** Convolutional Neural Network via **PyTorch** for fast, accurate image classification, and features a beautiful, dynamic frontend with modern glassmorphism UI.

---

## ✨ Features
- **Upload & Analyze**: Drag and drop leaf images for instant diagnosis.
- **Deep Learning Accuracy**: Employs transfer learning (MobileNetV2) trained on plant pathology datasets (e.g., PlantVillage).
- **Comprehensive Results**: Outputs disease name, confidence score, detailed description, and suggested remedies.
- **Beautiful UI/UX**: Responsive, dark-mode, animated interface providing a premium user experience.
- **Mock Fallback**: Backend gracefully falls back to mock predictions if a trained model weights file is not present.

---

## 🛠️ Tech Stack
**Frontend:**
- HTML5, CSS3 (Vanilla, Glassmorphism, CSS Animations)
- Vanilla JavaScript (Fetch API, Drag & Drop, File Reader API)

**Backend:**
- Python 3.9+
- FastAPI (REST API framework)
- Uvicorn (ASGI Server)

**Machine Learning / AI:**
- PyTorch (Deep Learning framework)
- Torchvision (Transforms, Pre-trained Models)
- Scikit-Learn (Metrics: Confusion Matrix, Classification Report)
- Matplotlib / Seaborn (Visualizations)

---

## 📁 Project Structure

```text
PhytoScan-AI/
│
├── backend/                  # FastAPI Application
│   ├── app.py                # Main API endpoints and model serving logic
│   ├── disease_info.json     # Database of plant diseases, descriptions, and remedies
│   └── requirements.txt      # Python dependencies for the backend
│
├── frontend/                 # Web Application UI
│   ├── index.html            # UI layout
│   ├── style.css             # Styling & Animations
│   └── script.js             # Logic for image upload and API integration
│
├── notebooks/                # Machine Learning Pipeline
│   └── train.py              # Script to train MobileNetV2 on a custom dataset
│
├── dataset/                  # Put your training dataset here (e.g. PlantVillage)
│
└── model/                    # Directory to store the trained .pth model files
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Techside-Pragyan/PhytoScan-AI.git
cd PhytoScan-AI
```

### 2. Backend Setup
Navigate to the backend directory and install the required Python packages:
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the API Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`. You can test endpoints at `http://localhost:8000/docs`.*

### 4. Run the Frontend
Simply open `frontend/index.html` in your web browser. Or, use a tool like Live Server or Python's built-in HTTP server:
```bash
cd frontend
python -m http.server 3000
```
*Access the app at `http://localhost:3000`.*

---

## 🧠 Model Training (Optional)

If you wish to train the model yourself instead of using the fallback:
1. Download a dataset like **PlantVillage**.
2. Organize it into folders under the `dataset/` directory (e.g., `dataset/Tomato_Early_blight/image.jpg`).
3. Run the training script:
   ```bash
   cd notebooks
   python train.py
   ```
4. The trained model will be saved as `model/plant_disease_model.pth`. Restart the FastAPI server to load the new weights.

---

## 📸 Screenshots
*(Add your screenshots here)*

---

## 🌐 Deployment
- **Backend**: Can be easily deployed on **Render** or **Railway** by setting the start command to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- **Frontend**: Deploy the contents of the `frontend/` folder directly to **Vercel**, **Netlify**, or **GitHub Pages**. (Ensure you update `API_URL` in `script.js` to point to your live backend domain).

---

> Built with ❤️ by an AI & Full-Stack Developer