import threading
from PIL import Image, ImageDraw
import face_recognition
import easygui


def show_message(text):
    easygui.textbox("Face Land Marks", "Landmark Details", text)


if __name__ == '__main__':
    image = face_recognition.load_image_file("img.png")

    face_landmarks_list = face_recognition.face_landmarks(image)
    pil_image = Image.fromarray(image)

    pil_image.show()

    if not face_landmarks_list:
        easygui.msgbox("No face landmarks detected!", title="Face Land Marks")
    else:
        face_landmarks_text = ""

        for face_landmarks in face_landmarks_list:
            for part, landmarks in face_landmarks.items():
                face_landmarks_text += f"{part}:\n"
                for x, y in landmarks:
                    face_landmarks_text += f"  ({x}, {y})\n"
            face_landmarks_text += "\n"

        thread = threading.Thread(target=show_message, args=(face_landmarks_text,))
        thread.start()

        for face_landmarks in face_landmarks_list:
            d = ImageDraw.Draw(pil_image, 'RGBA')
            if 'chin' in face_landmarks:
                d.line(face_landmarks['chin'], fill=(255, 255, 255, 255), width=3)
            if 'left_eyebrow' in face_landmarks:
                d.polygon(face_landmarks['left_eyebrow'], outline=(255, 255, 255, 255), width=3)
            if 'right_eyebrow' in face_landmarks:
                d.polygon(face_landmarks['right_eyebrow'], outline=(255, 255, 255, 255), width=3)
            if 'nose_bridge' in face_landmarks:
                d.polygon(face_landmarks['left_eye'], outline=(255, 255, 255, 255), width=3)
            if 'right_eye' in face_landmarks:
                d.polygon(face_landmarks['right_eye'], outline=(255, 255, 255, 255), width=3)
            if 'top_lip' in face_landmarks:
                d.polygon(face_landmarks['top_lip'], outline=(255, 255, 255, 255), width=3)
            if 'bottom_lip' in face_landmarks:
                d.polygon(face_landmarks['bottom_lip'], outline=(255, 255, 255, 255), width=3)

        pil_image.show()
