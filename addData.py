import os
import json
import tkinter as tk
from tkinter import *
from tkinter import filedialog

def select_directory():
    directory = filedialog.askdirectory()
    return directory

# Sélectionner le répertoire contenant les fichiers nécessaires
directory = select_directory()
if not directory:
    print("Vous devez sélectionner un répertoire.")
    exit()

intents_path = os.path.join(directory, 'intents.json')
icon_path = os.path.join(directory, 'suivi.ico')
script_path = os.path.join(directory, 'train_chatbot.py')

def add_data_to_json():
    new_intent = {
        "tag": entry_tag.get(),
        "patterns": entry_patterns.get().split(", "),
        "responses": entry_responses.get().split(", "),
        "image_path": entry_image_path.get(),  # Ajouter le champ image_path
        "context": []
    }
    with open(intents_path, "r") as f:
        data = json.load(f)
        data["intents"].append(new_intent)
    with open(intents_path, "w") as f:
        json.dump(data, f, indent=4)
    entry_tag.delete(0, tk.END)
    entry_patterns.delete(0, tk.END)
    entry_responses.delete(0, tk.END)
    entry_image_path.delete(0, tk.END)  # Effacer le champ image_path après l'ajout

root = tk.Tk()
root.title("Add Data")
root.iconbitmap(icon_path)
root.geometry("550x300")
root.resizable(width=FALSE, height=FALSE)
title = Label(root, text='Ajouter des données au chatbot', fg='gold', bg='black', font=('tajawal', 10, 'bold'))
title.pack(fill=X)

label_tag = tk.Label(root, text="Intent Tag:", fg='Black', font=('tajawal', 10, 'bold'))
label_tag.place(x=40, y=60)
entry_tag = tk.Entry(root, bg='white', fg='Black', width=40, font=('tajawal', 10, 'bold'))
entry_tag.place(x=250, y=50, height=40)

label_patterns = tk.Label(root, text="Patterns (Eviter virgules svp):", fg='Black', font=('tajawal', 10, 'bold'))
label_patterns.place(x=40, y=110)
entry_patterns = tk.Entry(root, bg='white', fg='Black', width=40, font=('tajawal', 10, 'bold'))
entry_patterns.place(x=250, y=100, height=40)

label_responses = tk.Label(root, text="Responses (Eviter virgules svp):", fg='Black', font=('tajawal', 10, 'bold'))
label_responses.place(x=40, y=160)
entry_responses = tk.Entry(root, bg='white', fg='Black', width=40, font=('tajawal', 10, 'bold'))
entry_responses.place(x=250, y=150, height=40)

label_image_path = tk.Label(root, text="Image Path:", fg='Black', font=('tajawal', 10, 'bold'))  # Champ pour l'image
label_image_path.place(x=40, y=210)
entry_image_path = tk.Entry(root, bg='white', fg='Black', width=40, font=('tajawal', 10, 'bold'))
entry_image_path.place(x=250, y=200, height=40)

import subprocess

def execute_script():
    subprocess.run(["python", script_path])  # Remplacez par le chemin réel du script Python

def testt():
    add_data_to_json()
    execute_script()

add_button = tk.Button(root, text="Ajouter", width=20, fg='black', bg='#DBA901', font=('tajawal', 11, 'bold'), command=testt)
add_button.place(x=180, y=260)

root.mainloop()
