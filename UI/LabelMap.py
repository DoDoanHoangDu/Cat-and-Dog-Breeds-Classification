class PetLabels(dict):
    def __init__(self):
        super().__init__()
        self.update({0: "Cat", 1: "Dog"})

class DogLabels(dict):
    def __init__(self):
        super().__init__()
        dog_breed_map = {0: 'n02085782-Japanese_spaniel', 1: 'n02085936-Maltese_dog', 2: 'n02086646-Blenheim_spaniel', 3: 'n02086910-papillon', 4: 'n02088094-Afghan_hound', 5: 'n02088466-bloodhound', 6: 'n02089078-black-and-tan_coonhound', 7: 'n02089973-English_foxhound', 8: 'n02091244-Ibizan_hound', 9: 'n02091467-Norwegian_elkhound', 10: 'n02092002-Scottish_deerhound', 11: 'n02092339-Weimaraner', 12: 'n02093647-Bedlington_terrier', 13: 'n02093754-Border_terrier', 14: 'n02093859-Kerry_blue_terrier', 15: 'n02095889-Sealyham_terrier', 16: 'n02096051-Airedale', 17: 'n02096294-Australian_terrier', 18: 'n02096437-Dandie_Dinmont', 19: 'n02096585-Boston_bull', 20: 'n02100236-German_short-haired_pointer', 21: 'n02101556-clumber', 22: 'n02102040-English_springer', 23: 'n02102177-Welsh_springer_spaniel', 24: 'n02102480-Sussex_spaniel', 25: 'n02104365-schipperke', 26: 'n02105505-komondor', 27: 'n02105641-Old_English_sheepdog', 28: 'n02105855-Shetland_sheepdog', 29: 'n02107312-miniature_pinscher', 30: 'n02107683-Bernese_mountain_dog', 31: 'n02108000-EntleBucher', 32: 'n02108422-bull_mastiff', 33: 'n02109525-Saint_Bernard', 34: 'n02110063-malamute', 35: 'n02110958-pug', 36: 'n02111129-Leonberg', 37: 'n02111889-Samoyed', 38: 'n02112018-Pomeranian', 39: 'n02112137-chow', 40: 'n02112350-keeshond', 41: 'n02112706-Brabancon_griffon', 42: 'n02113023-Pembroke', 43: 'n02113978-Mexican_hairless', 44: 'n02115913-dhole', 45: 'n02116738-African_hunting_dog'}
        self.update(dog_breed_map)
        

class CatLabels(dict):
    def __init__(self):
        super().__init__()
        cat_label_map = {0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 4: 'Egyptian Mau', 5: 'Exotic Shorthair', 6: 'Norwegian Forest', 7: 'Persian', 8: 'Russian Blue', 9: 'Scottish Fold', 10: 'Siamese', 11: 'Sphynx', 12: 'Turkish Angora'}
        self.update(cat_label_map)