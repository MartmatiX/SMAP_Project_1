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
    img = "img.png"

    # Upload the image into the context
    image = face_recognition.load_image_file(img)

    # Find the landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = Image.fromarray(image)

    # Show the original image
    pil_image.show()

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
                d.line(face_landmark['chin'], fill=(255, 255, 255, 255), width=3)
            if 'left_eyebrow' in face_landmark:
                d.polygon(face_landmark['left_eyebrow'], outline=(255, 255, 255, 255), width=3)
            if 'right_eyebrow' in face_landmark:
                d.polygon(face_landmark['right_eyebrow'], outline=(255, 255, 255, 255), width=3)
            if 'nose_bridge' in face_landmark:
                d.line(face_landmark['nose_bridge'], fill=(255, 255, 255, 255), width=3)
            if 'nose_tip' in face_landmark:
                d.line(face_landmark['nose_tip'], fill=(255, 255, 255, 255), width=3)
            if 'nose_bridge' in face_landmark:
                d.polygon(face_landmark['left_eye'], outline=(255, 255, 255, 255), width=3)
            if 'right_eye' in face_landmark:
                d.polygon(face_landmark['right_eye'], outline=(255, 255, 255, 255), width=3)
            if 'top_lip' in face_landmark:
                d.polygon(face_landmark['top_lip'], outline=(255, 255, 255, 255), width=3)
            if 'bottom_lip' in face_landmark:
                d.polygon(face_landmark['bottom_lip'], outline=(255, 255, 255, 255), width=3)
        # Show the image with the highlighted face landmarks
        pil_image.show()

        # Analyze the image - age, gender, race, emotion
        features = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender', 'race', 'emotion'],
        )

        age = ""
        gender_data = ""
        race_data = ""
        emotion_data = ""

        # Pick out respective values
        for feature in features:
            age = feature['age']
            gender_data = feature['gender']
            race_data = feature['race']
            emotion_data = feature['emotion']

        # Creating subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Gender chart
        axes[0, 0].bar(gender_data.keys(), gender_data.values(), color=['blue', 'green'])
        axes[0, 0].set_xlabel('Gender')
        axes[0, 0].set_ylabel('Probability (%)')
        axes[0, 0].set_title('Gender Distribution')
        axes[0, 0].set_ylim(0, 100)

        # Race chart
        axes[0, 1].bar(race_data.keys(), race_data.values(),
                       color=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
        axes[0, 1].set_xlabel('Race')
        axes[0, 1].set_ylabel('Probability (%)')
        axes[0, 1].set_title('Race Distribution')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Emotion chart
        axes[1, 0].bar(emotion_data.keys(), emotion_data.values(),
                       color=['pink', 'brown', 'cyan', 'lime', 'gray', 'magenta', 'black'])
        axes[1, 0].set_xlabel('Emotion')
        axes[1, 0].set_ylabel('Probability (%)')
        axes[1, 0].set_title('Emotion Distribution')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Age chart (single bar)
        axes[1, 1].bar(["Age"], [age], color='purple')
        axes[1, 1].set_xlabel('Feature')
        axes[1, 1].set_ylabel('Years')
        axes[1, 1].set_title('Estimated Age')
        axes[1, 1].set_ylim(0, 100)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the charts
        plt.show()
