import tkinter as tk
import os
import button_commands as command
import torch
from catdogmodel import CatDogModel
from catmodel import CatModel
from dogmodel import DogModel

dir_path = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.dirname(dir_path)
icon = os.path.join(dir_path,"UI\\sillycat.ico")


#main window
root = tk.Tk()
root.resizable(False, False)
root.title('Pet Identifier')
root.iconbitmap(icon)

#Label to display single/grid img
grid_frame = tk.Frame(root)

image_paths = []

def on_choose_image():
    global image_paths
    image_paths = command.choose_image(grid_frame)

def on_choose_random_images():
    global image_paths
    image_paths = command.choose_random_images(grid_frame)

button_single = tk.Button(root, text="Choose an Image", command=on_choose_image)
button_random = tk.Button(root, text="Choose Random Images", command=on_choose_random_images)



#load models
device = "cuda" if torch.cuda.is_available() else "cpu"

cd_model = CatDogModel(input_shape=3,
                        hidden_units=64,
                        output_shape=1,
                    dropout_prob = 0.5
                    ).to(device)
cd_model_path = os.path.join(dir_path,"CatDog_model_with_98.364%.pth") 
cd_model.load_state_dict(torch.load(cd_model_path,weights_only=True))
cd_model.eval()

cat_model = CatModel().to(device)
cat_model_path = os.path.join(dir_path,"Cat_model_with_90.659%.pth") 
cat_model.load_state_dict(torch.load(cat_model_path,weights_only=True))
cat_model.eval()

dog_model = DogModel().to(device)
dog_model_path = os.path.join(dir_path,"Dog_model_with_85.774%.pth") 
dog_model.load_state_dict(torch.load(dog_model_path,weights_only=True))
dog_model.eval()

def on_identify_cat_dog():
    if not image_paths or ("Pred" in command.get_label_text(grid_frame,0,0) and "Unknown" not in command.get_label_text(grid_frame,0,0)):
        on_choose_random_images()
    command.identify_cat_dog(image_paths,cd_model,device,grid_frame)
def on_identify_breed():
    if not image_paths:
        on_choose_random_images()
    if "Pred" not in command.get_label_text(grid_frame,0,0):
        on_identify_cat_dog()
    command.identify_breed(image_paths,[cat_model,dog_model],device,grid_frame)

button_prediction = tk.Button(root, text="Identify", command=on_identify_cat_dog)
button_breed = tk.Button(root, text="Identify Breed", command=on_identify_breed)

#layout
button_single.grid(row=0,column=0)
button_random.grid(row=0,column=1)
grid_frame.grid(row=1, column=0, columnspan=2)
button_prediction.grid(row = 2,column = 0)
button_breed.grid(row = 2, column=1)


# Start the main loop
root.mainloop()
