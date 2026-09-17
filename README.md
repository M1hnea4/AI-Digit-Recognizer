# 🧠 AI Digit Recognizer

An interactive Artificial Intelligence application that recognizes handwritten digits in real-time. Built with Python, this project uses a Neural Network (Multi-Layer Perceptron) trained on the classic MNIST dataset to predict user input.

## 🚀 Features
* **Interactive Canvas:** Draw digits directly on the screen using a mouse.
* **Live Predictions:** The AI processes the drawing, normalizes it, and outputs its prediction instantly.
* **Smart Memory:** The model trains itself on the full MNIST dataset (15,000 samples) on the first run and saves its "brain" (`.pkl` format) locally for instant loading on subsequent runs.
* **Image Processing:** Uses Pillow to resize, invert, and scale the drawn image to match the 28x28 pixel format of the training data.

## 🛠️ Technologies Used
* **Machine Learning:** `scikit-learn` (MLPClassifier)
* **GUI:** `Tkinter`
* **Image Processing & Data:** `Pillow` (PIL), `NumPy`, `Pandas`
* **Dataset:** MNIST (fetched via OpenML)

## 🎮 How to Run
1. Clone the repository and navigate to the project folder.
2. Install the required dependencies:
   \`\`\`bash
   pip install scikit-learn pillow numpy pandas
   \`\`\`
3. Run the application:
   \`\`\`bash
   python digit_recognizer.py
   \`\`\`
*(Note: The first run will take 1-2 minutes to download the dataset and train the model. Subsequent runs will load instantly).*
