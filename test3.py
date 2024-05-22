from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import os
# Load your pre-trained model for feature extraction
loaded_model = load_model('face_rec_acc_plus.h5')

# Load the OpenCV face detection model
pb_files = "/Users/peteranton/Desktop/peter/deep learning/lab6/opencv_face_detector_uint8.pb"
pbtxt_files = "/Users/peteranton/Desktop/peter/deep learning/lab6/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(pb_files, pbtxt_files)
person_names = ['Alex Lawther', 'Logan Lerman', 'Maria Pedraza', 'Anthony Mackie', 'Bobby Morley', 'Chris Evans', 'Chris Pratt', 
                 'Mark Zuckerberg', 'Anne Hathaway', 'Emilia Clarke', 'Cristiano Ronaldo', 'Josh Radnor', 'Henry Cavil', 'Zoe Saldana', 
                 'Ellen Page', 'Gwyneth Paltrow', 'Natalie Dormer', 'Barbara Palvin', 'Krysten Ritter', 'Elon Musk', 'Leonardo Dicaprio', 
                 'Bill Gates', 'Elizabeth Olsen', 'Megan Fox', 'Taylor Swift', 'Tom Hiddleston', 'Jeremy Renner', 'Melissa Fumero', 'Robert Downey Jr',
                   'Amber Heard', 'Jake Mcdorman', 'Robert De Niro', 'Grant Gustin', 'Jennifer Lawrence', 'Eliza Taylor', 'Scarlett Johansson', 
                   'Marie Avgeropoulos', 'Brenton Thwaites', 'Adriana Lima', 'Amanda Crew', 'Hugh Jackman', 'Katherine Langford', 'Camila Mendes', 'Selena Gomez', 
                   'Avril Lavigne', 'Jeff Bezos', 'Jimmy Fallon', 'Christian Bale', 'Shakira Isabel Mebarak', 'Alvaro Morte', 'Lindsey Morgan', 'Zac Efron', 
                   'Dominic Purcell', 'Barack Obama', 'Tom Holland', 'Emma Stone', 'Tom Hardy', 'Sophie Turner', 'Tuppence Middleton', 'Kiernen Shipka', 
                   'Andy Samberg', 'Elizabeth Lail', 'Ursula Corbero', 'Chris Hemsworth', 'Penn Badgley', 'Tom Cruise', 'Richard Harmon', 'Brian J. Smith',
                     'Neil Patrick Harris', 'Jason Momoa', 'Dwayne Johnson', 'Nadia Hilker', 'Wentworth Miller', 'Pedro Alonso', 'Stephen Amell', 
                     'Morena Baccarin', 'Inbar Lavi', 'Alycia Dabnem Carey', 'Zendaya', 'Madelaine Petsch', 'Mark Ruffalo', 'Alexandra Daddario', 
                     'Tom Ellis', 'Morgan Freeman', 'Natalie Portman', 'Emma Watson', 'Sarah Wayne Callies', 'Irina Shayk', 'Ben Affleck', 'Jessica Barden', 
                     'Brie Larson', 'Rebecca Ferguson', 'Maisie Williams', 'Millie Bobby Brown', 'Lionel Messi', 'Gal Gadot', 'Margot Robbie', 'Katharine Mcphee',
                       'Danielle Panabaker', 'Keanu Reeves', 'Johnny Depp', 'Miley Cyrus', 'Rami Malek', 'Rihanna', 'Lili Reinhart']
def detect_faces(net, frame):
    # Perform face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 177., 123.], False, False)
    net.setInput(blob)
    detections = net.forward()

    # Process the detections
    for i in range(0, detections.shape[2]):
        # Get the confidence (probability) of the current detection:
        confidence = detections[0, 0, i, 2]
        # Only consider detections if confidence is greater than a fixed minimum confidence:
        if confidence > 0.9:
            # Get the coordinates of the current detection:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the detection and the confidence:
            text = "{:.3f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)
            cropped_frame = frame[startY:endY, startX:endX]
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame,cropped_frame

def recognize_face_(unknown_face, model, no_class_dataset, person_names, threshold=0.05):
    resized_face = cv2.resize(unknown_face, (160, 160))
    preprocessed_face = preprocess_input(resized_face)
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
    
    predictions = model.predict(preprocessed_face)

    predicted_label = np.argmax(predictions)
    predicted_name = person_names[predicted_label]

    # Compute cosine similarity between predictions and all embeddings in the dataset
    similarities = cosine_similarity(predictions, no_class_dataset)

    # Find the maximum similarity
    max_similarity = np.max(similarities)

    # Check if the maximum similarity is above the threshold
    print(max_similarity)
    if max_similarity > threshold:
        return "Not Classified"
    else:
        return predicted_name
no_class_dataset = np.load('non_defined_features.npy')

if __name__ == '__main__':

    cap=cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cap.read()
            frame, cropped_frame = detect_faces(net, frame)
            inception_input = cv2.resize(cropped_frame, (160, 160))
            inception_input = inception_input / 255.0  
            image_to_predict = inception_input.reshape(1,160, 160, 3)
            prediction = recognize_face_(inception_input, loaded_model, no_class_dataset, person_names)
            cv2.putText(frame,prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)  
    cap.release()
    cv2.destroyAllWindows()
    