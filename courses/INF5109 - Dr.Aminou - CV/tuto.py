import cv2
import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import preprocess_input
from PIL import Image
from efficientnet.tfkeras import EfficientNetB0
# Charger le modèle pré-entraîné
model = EfficientNetB0(weights='imagenet')
input_shape = model.input_shape[1:3]

# Définir les classes d'objets que le modèle peut détecter
#with open('imagenet_classes.txt', 'r') as f:
    #classes = f.read().splitlines()
classes = """tench, Tinca tinca
goldfish, Carassius auratus
great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
tiger shark, Galeocerdo cuvieri
hammerhead, hammerhead shark
electric ray, crampfish, numbfish, torpedo
stingray
cock
hen
ostrich, Struthio camelus
brambling, Fringilla montifringilla
goldfinch, Carduelis carduelis
house finch, linnet, Carpodacus mexicanus
junco, snowbird
indigo bunting, indigo finch, indigo bird, Passerina cyanea
robin, American robin, Turdus migratorius
bulbul
jay
magpie
chickadee
humain
sac
machine portable
table
livre
stylo
ordinateur
chaise
tableau
calculatrice
tench
goldfish
great white shark
tiger shark
hammerhead
electric ray
stingray
cock
hen
ostrich
brambling
goldfinch
house finch
junco
indigo bunting
robin
bulbul
jay
magpie
chickadee
""".splitlines()

# Initialiser la webcam
cap = cv2.VideoCapture(0)

while True:
    print('000')
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionner l'image pour correspondre à la taille d'entrée du modèle
    resized_frame = cv2.resize(frame, input_shape)
    
    # Prétraiter l'image
    preprocessed_frame = preprocess_input(resized_frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Faire des prédictions avec le modèle
    predictions = model.predict(preprocessed_frame)
    class_index = np.argmax(predictions)
    class_label = classes[class_index]
    confidence = predictions[0][class_index]
    
    # Afficher les résultats de la détection sur l'image
    text = f'{class_label}: {confidence:.2f}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Object Detection', frame)
    print('####')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
