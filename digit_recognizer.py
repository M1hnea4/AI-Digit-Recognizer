import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import os
import pickle

class AIDigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Digit Recognizer - Pro Version")
        
        self.model_file = "ai_brain.pkl"
        
        # --- 1. ANTRENAMENTUL AI ---
        if os.path.exists(self.model_file):
            print("Încarc creierul AI-ului din memorie...")
            with open(self.model_file, 'rb') as f:
                self.ai_model = pickle.load(f)
            print("Gata!")
        else:
            print("Descarc datele MNIST. Așteaptă...")
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
            
            X_train, y_train = X[:15000], y[:15000]
            
            # REPARAT: Normalizare matematică pură (păstrează negrul ca valoare 0)
            X_train_scaled = X_train / 255.0
            
            print("Antrenez rețeaua neurală...")
            self.ai_model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, alpha=1e-4,
                                          solver='adam', random_state=1, learning_rate_init=.01)
            self.ai_model.fit(X_train_scaled, y_train)
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.ai_model, f)
            print("Antrenament complet și salvat!")

        # --- 2. INTERFAȚA GRAFICĂ ---
        self.canvas_width = 280
        self.canvas_height = 280
        
        self.lbl_info = tk.Label(self.root, text="Desenează o cifră (0-9) pe centru", font=("Arial", 12))
        self.lbl_info.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw_engine = ImageDraw.Draw(self.image)
        
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)
        
        self.btn_predict = tk.Button(self.btn_frame, text="🧠 Prezice", command=self.predict, bg="#2ecc71", fg="white", font=("Arial", 12, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(self.btn_frame, text="🗑️ Șterge", command=self.clear, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"))
        self.btn_clear.pack(side=tk.LEFT, padx=10)
        
        self.lbl_result = tk.Label(self.root, text="Aștept desenul...", font=("Arial", 16, "bold"), fg="#2c3e50")
        self.lbl_result.pack(pady=10)

    def draw(self, event):
        r = 12 # REPARAT: Pensulă adaptată la stilul MNIST
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw_engine.ellipse([x-r, y-r, x+r, y+r], fill="black")
        
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw_engine = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="Aștept desenul...", fg="#2c3e50")
        
    def predict(self):
        # 1. Redimensionăm și capturăm
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        
        # 2. Inversăm culorile (pentru că MNIST e antrenat pe fundal negru, scris alb)
        img_array = 255 - img_array
        
        # REPARAT 3: Aceeași formulă exactă ca la antrenament
        img_flattened = img_array.reshape(1, -1) / 255.0
        
        prediction = self.ai_model.predict(img_flattened)
        self.lbl_result.config(text=f"Sunt sigur că este: {prediction[0]}", fg="#2980b9")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIDigitRecognizer(root)
    root.mainloop()