from tensorflow.keras.models import load_model
import cv2
import numpy as np# Get the file path

pb_files = "opencv_face_detector_uint8.pb"
pbtxt_files = "opencv_face_detector.pbtxt"

# Function to perform face detection on a static photo
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

if "__main__" == __name__:
    loaded_model = load_model('face_rec_acc_plus.h5')
    net = cv2.dnn.readNetFromTensorflow(pb_files, pbtxt_files)
    cap = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cap.read()
            frame,cropped_frame = detect_faces(net, frame)            
           
            inception_input = cv2.resize(cropped_frame, (160, 160))
            inception_input = inception_input / 255.0  
            image_to_predict = inception_input.reshape(1,160, 160, 3)
            Y_pred = loaded_model.predict(image_to_predict)
            highest_prob_index = np.argmax(Y_pred)
            highest_prob_index
            person=person_names[highest_prob_index]
            x=Y_pred[0][highest_prob_index]
            if x<0.3:
                person="no class"
            cv2.putText(frame,person, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("frame",frame)
            print(person)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)

