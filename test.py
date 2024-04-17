from tensorflow.keras.models import load_model
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image, ImageDraw
import os
import tqdm
def image_gen():
    output_dir = "generated_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        image_size=(160,160)
        num_images = 70000
    for i in range(num_images):
        # Create a blank image with a white background
        image = Image.new("RGB", image_size, "white")
        
        # Create a drawing object
        draw = ImageDraw.Draw(image)
        
        # Randomize parameters for shape generation
        shape = np.random.choice(['rectangle', 'circle', 'triangle'])
        color = tuple(np.random.randint(0, 256, size=3))
        position = tuple(np.random.randint(0, size - 50) for size in image_size)
        size = np.random.randint(20, 80)
        
        # Draw a shape on the image
        if shape == 'rectangle':
            end_position = (position[0] + size, position[1] + size)
            draw.rectangle([position, end_position], fill=color)
        elif shape == 'circle':
            end_position = (position[0] + size, position[1] + size)
            draw.ellipse([position, end_position], fill=color)
        elif shape == 'triangle':
            points = [
                (position[0] + size // 2, position[1]),
                (position[0], position[1] + size),
                (position[0] + size, position[1] + size)
            ]
            draw.polygon(points, fill=color)
        
        # Save the image
        image_path = os.path.join(output_dir, f"image_{i}.png")
        image.save(image_path)

        if i % 1000 == 0:
            print(f"Generated {i}/{num_images} images")

    print("Image generation completed!")



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


# create image
image_gen()
#preprocess image and ransform to numpy array
non_defined_features = []
for i in tqdm.tqdm(range(70000)):
    image = Image.open(f"generated_images/image_{i}.png")
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = image.reshape(1, 160, 160, 3)
    non_defined_features.append(loaded_model.predict(image)[0])
non_defined_features.save('non_defined_features.npy')    


def detect_and_recognize_faces(net, frame, model, non_defined_features):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104., 177., 123.], False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cropped_frame = frame[startY:endY, startX:endX]
            
            # Resize and preprocess the cropped frame for prediction
            inception_input = cv2.resize(cropped_frame, (160, 160))
            inception_input = inception_input / 255.0  
            image_to_predict = inception_input.reshape(1, 160, 160, 3)
            
            # Predict using the loaded model
            Y_pred = model.predict(image_to_predict)
            highest_prob_index = np.argmax(Y_pred)
            highest_prob = Y_pred[0][highest_prob_index]
            
            # If highest probability is below threshold, classify as 'no class'
            if highest_prob < 0.4:
                person = "no class"
            else:
                # Calculate cosine similarity with non-classified faces
                features = model.predict(image_to_predict)[0]
                similarities = [1 - cosine(features, non_defined_features[i]) for i in range(len(non_defined_features))]
                
                # Define a threshold for classification
                threshold = 0.9
                
                # If any similarity exceeds threshold, classify as one of the known classes
                if max(similarities) > threshold:
                    highest_prob_index = np.argmax(similarities)
                    person = person_names[highest_prob_index]
                else:
                    person = "no class"

            cv2.putText(frame, person, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3)

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while True:
        try:
            ret, frame = cap.read()
            frame = detect_and_recognize_faces(net, frame, loaded_model, non_defined_features)
            
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print("Error:", e)

    cap.release()
    cv2.destroyAllWindows()
