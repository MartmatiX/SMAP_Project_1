import json

from PIL import Image, ImageDraw
from deepface import DeepFace

import threading
import face_recognition
import easygui
import matplotlib.pyplot as plt


# Separate method for textbox that is supplied to the thread
def show_message(msg, title, face_landmarks_text):
    easygui.textbox(msg, title, face_landmarks_text)


if __name__ == '__main__':
    # Define image path
    img = "img_3.png"

    # Upload the image into the context
    image = face_recognition.load_image_file(img)

    # Find the landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = Image.fromarray(image)

    # Show the original image
    pil_image.show()

    # Definition of known faces
    zelensky_image = face_recognition.load_image_file("known_faces/img.png")
    zelensky_face_encoding = face_recognition.face_encodings(zelensky_image)[0]

    pavel_image = face_recognition.load_image_file("known_faces/img_1.png")
    pavel_face_encoding = face_recognition.face_encodings(pavel_image)[0]

    obama_image = face_recognition.load_image_file("known_faces/img_2.png")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Definition of known encodings
    known_face_encodings = [
        zelensky_face_encoding,
        pavel_face_encoding,
        obama_face_encoding
    ]

    # Definition of known names
    known_face_names = [
        "Volodymyr Zelenskyj",
        "Petr Pavel",
        "Barack Obama"
    ]

    # Encode the image, so it can be used in the compare_faces method
    unknown_face_encodings = face_recognition.face_encodings(image)

    name = ""
    # Check if any faces were found
    if unknown_face_encodings:
        for unknown_encoding in unknown_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding)

            # Ensure there are known faces before calculating distances
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
                if len(face_distances) > 0:
                    best_match_index = face_distances.argmin()

                    if best_match_index < len(matches) and matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

    if not face_landmarks_list:
        easygui.msgbox("No face landmarks detected!", title="Face Land Marks")
    else:
        # Create face landmarks text for the message box
        face_landmarks_text = ""
        for face_landmark in face_landmarks_list:
            for part, landmarks in face_landmark.items():
                face_landmarks_text += f"{part}:\n"
                for x, y in landmarks:
                    face_landmarks_text += f"  ({x}, {y})\n"
            face_landmarks_text += "\n"

        # .textbox(args) is thread blocking, therefore dispatching separate thread for it
        thread = threading.Thread(target=show_message,
                                  args=("Face Land Marks", "Landmark Details", face_landmarks_text,))
        thread.start()

        # Highlight each found face mark
        for face_landmark in face_landmarks_list:
            d = ImageDraw.Draw(pil_image, 'RGBA')
            if 'chin' in face_landmark:
                d.line(face_landmark['chin'], fill=(255, 255, 255, 255), width=1)
            if 'left_eyebrow' in face_landmark:
                d.polygon(face_landmark['left_eyebrow'], outline=(255, 255, 255, 255), width=1)
            if 'right_eyebrow' in face_landmark:
                d.polygon(face_landmark['right_eyebrow'], outline=(255, 255, 255, 255), width=1)
            if 'nose_bridge' in face_landmark:
                d.line(face_landmark['nose_bridge'], fill=(255, 255, 255, 255), width=1)
            if 'nose_tip' in face_landmark:
                d.line(face_landmark['nose_tip'], fill=(255, 255, 255, 255), width=1)
            if 'nose_bridge' in face_landmark:
                d.polygon(face_landmark['left_eye'], outline=(255, 255, 255, 255), width=1)
            if 'right_eye' in face_landmark:
                d.polygon(face_landmark['right_eye'], outline=(255, 255, 255, 255), width=1)
            if 'top_lip' in face_landmark:
                d.polygon(face_landmark['top_lip'], outline=(255, 255, 255, 255), width=1)
            if 'bottom_lip' in face_landmark:
                d.polygon(face_landmark['bottom_lip'], outline=(255, 255, 255, 255), width=1)
        # Show the image with the highlighted face landmarks
        pil_image.show()

        # Analyze the image - age, gender, race, emotion
        features = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender', 'race', 'emotion'],
        )

        # Lists to store data for multiple faces
        all_ages = []
        all_genders = []
        all_races = []
        all_emotions = []
        dominant_data = []

        for i, feature in enumerate(features):
            all_ages.append(feature['age'])
            all_genders.append(feature['gender'])
            all_races.append(feature['race'])
            all_emotions.append(feature['emotion'])

            dominant_data.append({
                "Person": i + 1,
                "dominant_gender": feature['dominant_gender'],
                "dominant_race": feature['dominant_race'],
                "dominant_emotion": feature['dominant_emotion'],
                "estimated_age": feature['age'],
                "name": name
            })

        # Convert to JSON format for better readability
        json_data = json.dumps({"Detected People": dominant_data}, indent=4)

        # Dispatch separate thread and show the message
        thread2 = threading.Thread(target=show_message,
                                   args=("Detected People", "JSON output of detected people", json_data,))
        thread2.start()

        # Plotting the data
        num_people = len(features)
        fig, axes = plt.subplots(4, num_people, figsize=(4 * num_people, 16))
        if num_people == 1:
            axes = [[ax] for ax in axes]  # Ensure indexing works for a single person

        for i in range(num_people):
            # Gender chart
            axes[0][i].bar(all_genders[i].keys(), all_genders[i].values(), color=['blue', 'green'])
            axes[0][i].set_title(f'Gender - Person {i + 1}')

            # Race chart
            axes[1][i].bar(all_races[i].keys(), all_races[i].values(),
                           color=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
            axes[1][i].set_title(f'Race - Person {i + 1}')

            # Emotion chart
            axes[2][i].bar(all_emotions[i].keys(), all_emotions[i].values(),
                           color=['pink', 'brown', 'cyan', 'lime', 'gray', 'magenta', 'black'])
            axes[2][i].set_title(f'Emotion - Person {i + 1}')

            # Age chart
            axes[3][i].bar(["Age"], [all_ages[i]], color='purple')
            axes[3][i].set_title(f'Estimated Age - Person {i + 1}')

        plt.tight_layout()
        plt.show()
