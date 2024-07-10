import os
import face_recognition_api
import pickle
import numpy as np
import pandas as pd
import cv2

def get_prediction_images(prediction_dir):
    files = [x[2] for x in os.walk(prediction_dir)][0]
    l = []
    exts = [".jpg", ".jpeg", ".png"]
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in exts:
            l.append(os.path.join(prediction_dir, file))
    return l

fname = 'classifier.pkl'
prediction_dir = './test-images'

encoding_file_path = './encoded-images-data.csv'
df = pd.read_csv(encoding_file_path)
full_data = np.array(df.astype(float).values.tolist())

# Extract features and labels
X = np.array(full_data[:, 1:-1])
y = np.array(full_data[:, -1:])

if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        (le, clf) = pickle.load(f)
else:
    print('\x1b[0;37;43m' + "Classifier '{}' does not exist".format(fname) + '\x1b[0m')
    quit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame
    X_faces_loc = face_recognition_api.face_locations(img)

    faces_encodings = face_recognition_api.face_encodings(img, known_face_locations=X_faces_loc)
    print("Found {} faces in the image".format(len(faces_encodings)))

    if faces_encodings:
        closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(X_faces_loc))]

        # Ensure predictions are made as a 1-dimensional array
        predictions = clf.predict(faces_encodings)
        if isinstance(predictions, np.ndarray) and predictions.ndim > 0:
            predictions = predictions.ravel()

        # Debugging print statements
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions: {predictions}")

        # Perform inverse transformation
        try:
            predictions = [(le.inverse_transform([int(pred)])[0].title(), loc) if rec else ("Unknown", loc)
                           for pred, loc, rec in zip(predictions, X_faces_loc, is_recognized)]
        except ValueError as e:
            print(f"Error in inverse_transform: {e}")

        print(predictions)
        print()

        # Display the resulting frame with annotations
        for (name, (top, right, bottom, left)) in predictions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        print("No faces found in the image")

    cv2.imshow('Video', frame)

    # Check for user input to quit (press 'q' key or close window)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('x'):  # Check for 'x' key press (optional)
        break

    # Check if the user has closed the window
    if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
