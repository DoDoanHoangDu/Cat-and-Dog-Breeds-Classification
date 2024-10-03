
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from collections import defaultdict


class CNNClassification(nn.Module):
    def __init__(self):
        super(CNNClassification, self).__init__()

        self.CNN_Model = nn.Sequential(
            # Block 1: Two Conv layers + Pooling
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Pooling

            # Block 2: Two Conv layers + Pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Pooling

            # Block 3: Three Conv layers + Pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Pooling

            # Block 4: Three Conv layers + Pooling
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Pooling

            nn.Flatten(),

            # Fully connected layers with Dropout
            nn.Linear(256 * 9 * 9, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output layer for 120 classes
            nn.Linear(1024, 120)
        )

    def forward(self, x):
        return self.CNN_Model(x)

model = CNNClassification()  

model.load_state_dict(torch.load('DogBreed_90_53percent_top5_2-10-2024.pth'))

model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



def inference(path, model_chosen, device):
    
    r = requests.get(path)
    with BytesIO(r.content) as f:
        img = Image.open(f).convert("RGB")  
        img = img.resize((150,150))  

    
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    img_tensor = transform(img).unsqueeze(0)  
    
    
    img_tensor = img_tensor.to(device)

    
    model_chosen.eval()

    with torch.no_grad(): 
        predictions = model_chosen(img_tensor)
        probabilities = torch.softmax(predictions, dim=1) 
        predicted_class = torch.argmax(probabilities, dim=1).item()  

    return predicted_class, probabilities.cpu().numpy()  


#Load image from internet . Go find image you want -> right click to get the link to image -> paste it here
path = "https://thehappypuppysite.com/wp-content/uploads/2017/12/siberian6.jpg"
r = requests.get(path)
with BytesIO(r.content) as f:
    img = Image.open(f)
    img = img.resize((150,150))
    x = np.array(img) / 255.0



predicted_value, probability = inference(path= path, model_chosen= model,device= device)
print(predicted_value)



breed_map = {0: 'Chihuahua', 
             1: 'Japanese_spaniel', 
             2: 'Maltese_dog', 
             3: 'Pekinese', 
             4: 'Shih-Tzu', 
             5: 'Blenheim_spaniel', 
             6: 'papillon', 
             7: 'toy_terrier', 
             8: 'Rhodesian_ridgeback', 
             9: 'Afghan_hound', 
             10: 'basset', 
             11: 'beagle', 
             12: 'bloodhound', 
             13: 'bluetick', 
             14: 'black-and-tan_coonhound', 
             15: 'Walker_hound', 
             16: 'English_foxhound', 
             17: 'redbone', 
             18: 'borzoi', 
             19: 'Irish_wolfhound', 
             20: 'Italian_greyhound', 
             21: 'whippet', 
             22: 'Ibizan_hound', 
             23: 'Norwegian_elkhound', 
             24: 'otterhound', 
             25: 'Saluki', 
             26: 'Scottish_deerhound', 
             27: 'Weimaraner', 
             28: 'Staffordshire_bullterrier', 
             29: 'American_Staffordshire_terrier', 
             30: 'Bedlington_terrier', 
             31: 'Border_terrier', 
             32: 'Kerry_blue_terrier', 
             33: 'Irish_terrier', 
             34: 'Norfolk_terrier', 
             35: 'Norwich_terrier', 
             36: 'Yorkshire_terrier', 
             37: 'wire-haired_fox_terrier', 
             38: 'Lakeland_terrier', 
             39: 'Sealyham_terrier', 
             40: 'Airedale', 
             41: 'cairn', 
             42: 'Australian_terrier', 
             43: 'Dandie_Dinmont', 
             44: 'Boston_bull', 
             45: 'miniature_schnauzer', 
             46: 'giant_schnauzer', 
             47: 'standard_schnauzer', 
             48: 'Scotch_terrier', 
             49: 'Tibetan_terrier', 
             50: 'silky_terrier', 
             51: 'soft-coated_wheaten_terrier', 
             52: 'West_Highland_white_terrier', 
             53: 'Lhasa', 
             54: 'flat-coated_retriever', 
             55: 'curly-coated_retriever', 
             56: 'golden_retriever', 
             57: 'Labrador_retriever', 
             58: 'Chesapeake_Bay_retriever', 
             59: 'German_short-haired_pointer', 
             60: 'vizsla', 
             61: 'English_setter', 
             62: 'Irish_setter', 
             63: 'Gordon_setter', 
             64: 'Brittany_spaniel', 
             65: 'clumber', 
             66: 'English_springer', 
             67: 'Welsh_springer_spaniel', 
             68: 'cocker_spaniel', 
             69: 'Sussex_spaniel', 
             70: 'Irish_water_spaniel', 
             71: 'kuvasz', 
             72: 'schipperke', 
             73: 'groenendael', 
             74: 'malinois', 
             75: 'briard', 
             76: 'kelpie', 
             77: 'komondor', 
             78: 'Old_English_sheepdog', 
             79: 'Shetland_sheepdog', 
             80: 'collie', 
             81: 'Border_collie', 
             82: 'Bouvier_des_Flandres', 
             83: 'Rottweiler', 
             84: 'German_shepherd', 
             85: 'Doberman', 
             86: 'miniature_pinscher', 
             87: 'Greater_Swiss_Mountain_dog', 
             88: 'Bernese_mountain_dog', 
             89: 'Appenzeller', 
             90: 'EntleBucher', 
             91: 'boxer', 
             92: 'bull_mastiff', 
             93: 'Tibetan_mastiff', 
             94: 'French_bulldog', 
             95: 'Great_Dane', 
             96: 'Saint_Bernard', 
             97: 'Eskimo_dog', 
             98: 'malamute', 
             99: 'Siberian_husky', 
             100: 'affenpinscher', 
             101: 'basenji', 
             102: 'pug', 
             103: 'Leonberg', 
             104: 'Newfoundland', 
             105: 'Great_Pyrenees', 
             106: 'Samoyed', 
             107: 'Pomeranian', 
             108: 'chow', 
             109: 'keeshond', 
             110: 'Brabancon_griffon', 
             111: 'Pembroke', 
             112: 'Cardigan', 
             113: 'toy_poodle', 
             114: 'miniature_poodle', 
             115: 'standard_poodle', 
             116: 'Mexican_hairless',
             117: 'dingo', 
             118: 'dhole', 
             119: 'African_hunting_dog'}


print("Predicted breed : ",breed_map[predicted_value])

plt.imshow(x)
plt.title(f"Dog breed : {breed_map[predicted_value]}")
plt.axis('off')
plt.show()

