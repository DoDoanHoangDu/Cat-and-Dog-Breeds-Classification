import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
IMG_SIZE = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_units = 64
dir_path = os.path.dirname(os.path.abspath(__file__))


class CatModel(nn.Module):
    def __init__(self):
        super(CatModel, self).__init__()
        #Block 1
        self.layer1 = nn.Sequential(

            nn.Conv2d(3, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU())
        self.layer2 = nn.Sequential(

            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 2
        self.layer3 = nn.Sequential(

            nn.Conv2d(hidden_units, hidden_units*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU())
        self.layer4 = nn.Sequential(

            nn.Conv2d(hidden_units*2, hidden_units*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 3
        self.layer5 = nn.Sequential(

            nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU())
        self.layer6 = nn.Sequential(

            nn.Conv2d(hidden_units*4, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.Conv2d(hidden_units*4, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 4
        self.layer7 = nn.Sequential(

            nn.Conv2d(hidden_units*4, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU())
        self.layer8 = nn.Sequential(

            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 5
        self.layer9 = nn.Sequential(

            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU())
        self.layer10 = nn.Sequential(

            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*hidden_units*8, 8192),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 8192),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 13))






    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

class DogModel(nn.Module):
    def __init__(self):
        super(DogModel, self).__init__()
        #Block 1
        self.layer1 = nn.Sequential(

            nn.Conv2d(3, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU())
        self.layer2 = nn.Sequential(

            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 2
        self.layer3 = nn.Sequential(

            nn.Conv2d(hidden_units, hidden_units*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU())
        self.layer4 = nn.Sequential(

            nn.Conv2d(hidden_units*2, hidden_units*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 3
        self.layer5 = nn.Sequential(

            nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU())
        self.layer6 = nn.Sequential(

            nn.Conv2d(hidden_units*4, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.Conv2d(hidden_units*4, hidden_units*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 4
        self.layer7 = nn.Sequential(

            nn.Conv2d(hidden_units*4, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU())
        self.layer8 = nn.Sequential(

            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Block 5
        self.layer9 = nn.Sequential(

            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU())
        self.layer10 = nn.Sequential(

            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.Conv2d(hidden_units*8, hidden_units*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        #Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*hidden_units*8, 8192),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 8192),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8192, 46))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

class CatDogModel(nn.Module):
    def __init__(self,input_shape, hidden_units, output_shape,dropout_prob = 0.2):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels = hidden_units,
                      out_channels = hidden_units,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels = hidden_units*2,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.Conv2d(in_channels = hidden_units*2,
                      out_channels = hidden_units*2,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*2),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*2,
                      out_channels = hidden_units*4,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.Conv2d(in_channels = hidden_units*4,
                      out_channels = hidden_units*4,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.Conv2d(in_channels = hidden_units*4,
                      out_channels = hidden_units*4,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*4),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*4,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.Conv2d(in_channels = hidden_units*8,
                      out_channels = hidden_units*8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units*8),
            nn.MaxPool2d(kernel_size = 2)
        )
        flatten_size = self.initialize_classifier(input_shape)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=flatten_size,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=4096,
                      out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=4096,
                      out_features=output_shape),
        )

    def initialize_classifier(self, input_shape):
        dummy_input = torch.zeros(1, input_shape, IMG_SIZE, IMG_SIZE)
        dummy_output = self.conv_block_1(dummy_input)
        dummy_output = self.conv_block_2(dummy_output)
        dummy_output = self.conv_block_3(dummy_output)
        dummy_output = self.conv_block_4(dummy_output)
        dummy_output = self.conv_block_5(dummy_output)
        flatten_size = dummy_output.numel()
        return flatten_size

    def forward(self,x):
        x = self.conv_block_1(x)
        x= self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.classifier(x)
        return x


cd_model = CatDogModel(input_shape=3,
                        hidden_units=64,
                        output_shape=1,
                    dropout_prob = 0.5
                    ).to(device)
cd_model_path = os.path.join(dir_path,"CatDog_model_with_98.364%.pth") 
print("Loading cat dog model ...")
cd_model.load_state_dict(torch.load(cd_model_path,weights_only=True))
cd_model.eval()
print("Model loaded.")

catdog_map = {0: "Cat", 1: "Dog"}
dog_breed_map = {0: 'n02085782-Japanese_spaniel', 1: 'n02085936-Maltese_dog', 2: 'n02086646-Blenheim_spaniel', 3: 'n02086910-papillon', 4: 'n02088094-Afghan_hound', 5: 'n02088466-bloodhound', 6: 'n02089078-black-and-tan_coonhound', 7: 'n02089973-English_foxhound', 8: 'n02091244-Ibizan_hound', 9: 'n02091467-Norwegian_elkhound', 10: 'n02092002-Scottish_deerhound', 11: 'n02092339-Weimaraner', 12: 'n02093647-Bedlington_terrier', 13: 'n02093754-Border_terrier', 14: 'n02093859-Kerry_blue_terrier', 15: 'n02095889-Sealyham_terrier', 16: 'n02096051-Airedale', 17: 'n02096294-Australian_terrier', 18: 'n02096437-Dandie_Dinmont', 19: 'n02096585-Boston_bull', 20: 'n02100236-German_short-haired_pointer', 21: 'n02101556-clumber', 22: 'n02102040-English_springer', 23: 'n02102177-Welsh_springer_spaniel', 24: 'n02102480-Sussex_spaniel', 25: 'n02104365-schipperke', 26: 'n02105505-komondor', 27: 'n02105641-Old_English_sheepdog', 28: 'n02105855-Shetland_sheepdog', 29: 'n02107312-miniature_pinscher', 30: 'n02107683-Bernese_mountain_dog', 31: 'n02108000-EntleBucher', 32: 'n02108422-bull_mastiff', 33: 'n02109525-Saint_Bernard', 34: 'n02110063-malamute', 35: 'n02110958-pug', 36: 'n02111129-Leonberg', 37: 'n02111889-Samoyed', 38: 'n02112018-Pomeranian', 39: 'n02112137-chow', 40: 'n02112350-keeshond', 41: 'n02112706-Brabancon_griffon', 42: 'n02113023-Pembroke', 43: 'n02113978-Mexican_hairless', 44: 'n02115913-dhole', 45: 'n02116738-African_hunting_dog'}
cat_label_map = {0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 4: 'Egyptian Mau', 5: 'Exotic Shorthair', 6: 'Norwegian Forest', 7: 'Persian', 8: 'Russian Blue', 9: 'Scottish Fold', 10: 'Siamese', 11: 'Sphynx', 12: 'Turkish Angora'}
map_map = {'Cat': cat_label_map, 'Dog': dog_breed_map}

def load_model(type):
    if type == 'Cat':
        model = CatModel().to(device)
        model_path = os.path.join(dir_path,"Cat_model_with_90.659%.pth") 
    elif type == 'Dog':
        model = DogModel().to(device)
        model_path = os.path.join(dir_path,"Dog_model_with_85.774%.pth")
    print(f"Loading {type} Breeds model ...") 
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    print("Model loaded.")
    return model

def preprocess_image(image_path, img_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def predict(image_path, model, class_names,catdog = False):
    image_tensor = preprocess_image(image_path)
    with torch.inference_mode():
        outputs = model(image_tensor)
        if catdog:
            predictions = torch.round(torch.sigmoid(outputs)).squeeze()
        else:
            predictions = torch.argmax(torch.softmax(outputs,dim=1))

    return class_names[predictions.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Inference Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the test image")
    args = parser.parse_args()

    type = predict(args.image_path, cd_model, catdog_map,catdog=True)
    print(f"Type: {type}")
    result = predict(args.image_path, load_model(type), map_map[type])

    # Output the classification result
    print(f"Type: {type}")
    print(f"Breeds: {result}")
