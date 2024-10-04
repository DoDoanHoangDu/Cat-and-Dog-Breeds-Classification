import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from LabelMap import PetLabels, DogLabels,CatLabels

dir_path = os.path.dirname(os.path.abspath(__file__))
FOLDER_CATS = os.path.join(dir_path, "catbreeds\\images")
FOLDER_DOGS = os.path.join(dir_path, "dogbreeds\\Images")

classes = PetLabels()
dog_classes = DogLabels()
for key, breed in dog_classes.items():
    breed = breed.split("_")
    breed = " ".join(breed)
    breed = breed.title()
    dog_classes[key] = breed
cat_classes = CatLabels()

dog_names = set(dog_classes.values())
cat_names = set(cat_classes.values())

def choose_image(grid_frame):
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*"))
    )
    if file_path:
        update_image_frame([[file_path,"Unknown"]],grid_frame)
    return [[file_path,"Unknown"]]



def choose_random_images(grid_frame, folders = [FOLDER_CATS, FOLDER_DOGS]):
    chosen_images = []
    image_set = set()
    while len(chosen_images) < 9:
        folder = random.choice(folders)
        if os.path.exists(folder):
            subfolders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            subfolder = random.choice(subfolders)

            image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            random_image = random.choice(image_files)
            cd = "Cat - " if "catbreeds" in subfolder else "Dog - "
            breed = subfolder.split("\\")[-1]
            if cd == "Dog - ":
                breed = breed.split("-")
                breed.pop(0)
                breed = "_".join(breed)
                breed = breed.split("_")
                breed = " ".join(breed)
                breed = breed.title()
            label = cd+breed
            if (cd == "Dog - " and breed not in dog_names) or (cd == "Cat - " and breed not in cat_names):
                continue
            print(label)
            if random_image not in image_set:
                image_set.add(random_image)
                chosen_images.append([random_image,label])
    update_image_frame(chosen_images,grid_frame)
    return chosen_images


def update_image_frame(chosen_images,grid_frame,pred_label = None):
    for widget in grid_frame.winfo_children():
        widget.destroy()
    for index, [image_path,label] in enumerate(chosen_images):
        try:
            img = Image.open(image_path)
            if len(chosen_images) == 9:
                img = img.resize((150, 150), Image.Resampling.LANCZOS)
            elif len(chosen_images) == 1:
                img = img.resize((500, 500), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            img_frame = tk.Frame(grid_frame)
            img_label = tk.Label(img_frame, image=img_tk)
            img_label.image = img_tk  #reference to avoid garbage collection
            img_label.pack()
            color = "black"
            if pred_label:
                if label != "Unknown":
                    color = "green" if pred_label[index] in label else "red"
                label = "Pred: " + pred_label[index] + "\nTruth: " + label
            text_label = tk.Label(img_frame, text=label, font=("Arial", 10), fg=color)
            text_label.pack()

            img_frame.grid(row=index // 3, column=index % 3, padx=5, pady=5)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    print("-----------------------------------------------------------------------------------------------------------------------------")



def identify_cat_dog(image_paths,model,device,grid_frame):
    if image_paths:
        predicted_classes = []
        images = [i[0] for i in image_paths]
        labels = [i[1] for i in image_paths]
        transformed_images = []
        transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        for path in images:
            img = Image.open(path).convert('RGB')
            img = transform(img)  # Apply transformations
            transformed_images.append(img)
        images_tensor = torch.stack(transformed_images)
        dataloader = DataLoader(images_tensor, batch_size=1, shuffle=False)
        model.eval()
        with torch.inference_mode():
            for images in dataloader:
                images = images.to(device)
                logits = model(images).squeeze()
                print(torch.sigmoid(logits))
                predictions = torch.round(torch.sigmoid(logits)).squeeze()
                predicted_classes.append(classes[predictions.item()])
            print(predicted_classes)
        update_image_frame(image_paths,grid_frame,predicted_classes)

def get_label_text(grid_frame, row, col):
    widgets_in_position = grid_frame.grid_slaves(row=row, column=col)
    frame = widgets_in_position[0]
    for child in frame.winfo_children():
        if isinstance(child, tk.Label):
            text = child.cget("text")
            if text:
                return text
    return None

def identiy_breed(image_paths,models,device,grid_frame):
    model_cats , model_dogs = models
    cat_image_index = []
    dog_image_index = []
    if image_paths:
        if "Pred" in get_label_text(grid_frame,0,0) :
            num_image = len(image_paths)
            nrow = int(num_image**0.5)
            for row in range(nrow):
                for col in range(nrow):
                    label_text = get_label_text(grid_frame,row,col)
                    print(label_text)
                    if "Pred: Cat" in label_text:
                        cat_image_index.append(row*nrow+col)
                    else:
                        dog_image_index.append(row*nrow+col)
            predicted_classes = ["" for i in range(num_image)]
            labels = [i[1] for i in image_paths]
            transform = transforms.Compose([transforms.Resize((150, 150)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            if cat_image_index:
                cat_images = [image_paths[index] for index in cat_image_index]
                transformed_cat_images = []
                for path in cat_images:
                    img = Image.open(path[0]).convert('RGB')
                    img = transform(img)  # Apply transformations
                    transformed_cat_images.append(img)
                cat_images_tensor = torch.stack(transformed_cat_images)
                cat_dataloader = DataLoader(cat_images_tensor, batch_size=1, shuffle=False)
                model_cats.eval()
                with torch.inference_mode():
                    for index,images in enumerate(cat_dataloader):
                        images = images.to(device)
                        logits = model_cats(images)
                        predictions = torch.argmax(torch.softmax(logits,dim=1))
                        predicted_classes[cat_image_index[index]] = "Cat - " + cat_classes[predictions.item()]
            if dog_image_index:
                dog_images = [image_paths[index] for index in dog_image_index]
                transformed_dog_images = []
                for path in dog_images:
                    img = Image.open(path[0]).convert('RGB')
                    img = transform(img)  # Apply transformations
                    transformed_dog_images.append(img)
                dog_images_tensor = torch.stack(transformed_dog_images)
                dog_dataloader = DataLoader(dog_images_tensor, batch_size=1, shuffle=False)
                model_dogs.eval()
                with torch.inference_mode():
                    for index,images in enumerate(dog_dataloader):
                        images = images.to(device)
                        logits = model_dogs(images)
                        predictions = torch.argmax(torch.softmax(logits,dim=1))
                        predicted_classes[dog_image_index[index]] = "Dog - " + dog_classes[predictions.item()]
            print(predicted_classes)
            update_image_frame(image_paths,grid_frame,predicted_classes)
                