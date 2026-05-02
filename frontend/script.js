document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultSection = document.getElementById('result-section');
    const resetBtn = document.getElementById('reset-btn');
    
    // Result elements
    const diseaseName = document.getElementById('disease-name');
    const confidenceValue = document.getElementById('confidence-value');
    const diseaseDesc = document.getElementById('disease-desc');
    const diseaseRemedy = document.getElementById('disease-remedy');
    
    let currentFile = null;
    
    // API URL (Update this when deploying)
    const API_URL = 'http://localhost:8000/predict';

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('drag-over'), false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle File Input Change
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            dropArea.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.classList.remove('hidden');
            resultSection.classList.add('hidden');
        }
    }

    // Remove Image
    removeBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        dropArea.classList.remove('hidden');
        resultSection.classList.add('hidden');
    });

    // Reset Flow
    resetBtn.addEventListener('click', () => {
        removeBtn.click();
    });

    // Analyze Image API Call
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State -> Loading
        analyzeBtn.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            if (data.success) {
                // Populate results
                diseaseName.textContent = data.disease_name;
                confidenceValue.textContent = data.confidence;
                diseaseDesc.textContent = data.description;
                diseaseRemedy.textContent = data.remedy;
                
                // Switch UI State -> Result
                loadingState.classList.add('hidden');
                resultSection.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'Prediction failed');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze image. Ensure backend is running at ' + API_URL);
            loadingState.classList.add('hidden');
            analyzeBtn.classList.remove('hidden');
        }
    });
});
