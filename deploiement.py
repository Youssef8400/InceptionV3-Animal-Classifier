import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.applications.inception_v3 import preprocess_input
from keras._tf_keras.keras.preprocessing import image

MODEL_PATH = "animal_classifier_inceptionv3.h5"
IMG_SIZE = (299, 299)
DISPLAY_SIZE = (200, 200) 

model = load_model(MODEL_PATH)

class_labels = [
    'butterfly', 'cat', 'chicken', 'cow', 'dog',
    'elephant', 'horse', 'sheep', 'spider', 'squirrel'
]

root = tk.Tk()
root.title("Prédiction d'Animal 🐾")
root.geometry("600x303")
root.resizable(False, False)
root.configure(bg="white")

main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

left_frame = tk.Frame(main_frame, bg="#f0f4ff", width=200, height=250, highlightthickness=2, highlightbackground="#5c7cfa")
left_frame.pack(side="left", fill="y", padx=5, pady=5)
left_frame.pack_propagate(False)

upload_label = tk.Label(left_frame, text="📂 Upload Image", font=("Helvetica", 12, "bold"), bg="#f0f4ff", fg="#34495e")
upload_label.pack(pady=(30, 10))

browse_btn = tk.Button(
    left_frame,
    text="Browse...",
    font=("Helvetica", 11, "bold"),
    bg="#5c7cfa",
    fg="white",
    relief="flat",
    width=12,
    command=lambda: load_image()
)
browse_btn.pack(pady=10)

right_frame = tk.Frame(main_frame, bg="white")
right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

image_frame = tk.Frame(right_frame, bg="#f8f9fa", highlightbackground="#ccc", highlightthickness=1)
image_frame.pack(pady=10)
image_panel = tk.Label(image_frame, bg="#f8f9fa", width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])
image_panel.pack()

result_frame = tk.Frame(right_frame, bg="#e8f5e9", highlightbackground="#2ecc71", highlightthickness=1)
result_frame.pack(pady=10, fill="x")
result_label = tk.Label(result_frame, text="Aucune image chargée", font=("Helvetica", 14, "bold"), fg="#2e7d32", bg="#e8f5e9")
result_label.pack(pady=5)

last_image_path = None

def load_image():
    global last_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    last_image_path = file_path
    display_image(file_path)
    predict_image(file_path)

def display_image(file_path):
    img = Image.open(file_path).convert("RGB").resize(DISPLAY_SIZE)
    tk_img = ImageTk.PhotoImage(img)
    image_panel.configure(image=tk_img)
    image_panel.image = tk_img

def predict_image(file_path):
    img = Image.open(file_path).convert("RGB").resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    predicted_class = class_labels[pred_index]
    confidence = preds[0][pred_index] * 100
    result_label.config(text=f"{predicted_class} ({confidence:.2f}%)")

root.mainloop()
