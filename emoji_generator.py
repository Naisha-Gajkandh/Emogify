import cv2
import numpy as np

# Map model classes to explicit emotion labels
EMOTION_MAP = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "shocked",
    5: "confused",
    6: "excited",
    7: "disgusted",
    8: "fearful",
    9: "laugh",
    10: "surprised"
}

def draw_emoji(gender, emotion_class, size=(150, 150)):
    """Dynamically draws an emoji as a BGRA numpy array based on gender and emotion."""
    img = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    
    cx, cy = size[0] // 2, size[1] // 2
    r = int(min(size) * 0.45)
    
    # 1. Base Face (Yellow)
    cv2.circle(img, (cx, cy), r, (0, 200, 255, 255), -1)
    # Face outline
    cv2.circle(img, (cx, cy), r, (0, 150, 200, 255), 2)

    # 2. Gender specific features
    if gender == 'female':
        # Draw a pink bow on top right
        bow_color = (150, 50, 255, 255) # BGR pink
        cv2.fillPoly(img, [np.array([[cx+20, cy-r], [cx+40, cy-r-15], [cx+40, cy-r+5]])], bow_color)
        cv2.fillPoly(img, [np.array([[cx+20, cy-r], [cx, cy-r-15], [cx, cy-r+5]])], bow_color)
        cv2.circle(img, (cx+20, cy-r), 5, (100, 0, 200, 255), -1)
        # Eyelashes will be added below
    elif gender == 'male':
        # Draw short hair or a subtle cap
        cap_color = (250, 50, 50, 255) # Blue cap
        cv2.ellipse(img, (cx, cy-r+10), (int(r*0.9), int(r*0.3)), 0, 180, 360, cap_color, -1)
        cv2.ellipse(img, (cx, cy-r+10), (int(r*0.9), int(r*0.3)), 0, 180, 360, (150, 0, 0, 255), 2)
        # cap brim
        cv2.line(img, (cx-int(r*0.9), cy-r+10), (cx+int(r*1.1), cy-r+10), cap_color, 4)

    emotion = EMOTION_MAP.get(emotion_class, "neutral")

    # 3. Eyes
    eye_offset_x = int(r * 0.35)
    eye_offset_y = int(r * 0.2)
    lx, ly = cx - eye_offset_x, cy - eye_offset_y
    rx, ry = cx + eye_offset_x, cy - eye_offset_y
    
    if emotion in ["happy", "excited", "laugh"]:
        # Curved happy eyes ^^
        cv2.ellipse(img, (lx, ly), (15, 10), 0, 180, 360, (0, 0, 0, 255), 3)
        cv2.ellipse(img, (rx, ry), (15, 10), 0, 180, 360, (0, 0, 0, 255), 3)
    elif emotion in ["sad", "fearful", "disgusted"]:
        # Sad eyes
        cv2.ellipse(img, (lx, ly+5), (15, 10), 0, 0, 180, (0, 0, 0, 255), 3)
        cv2.ellipse(img, (rx, ry+5), (15, 10), 0, 0, 180, (0, 0, 0, 255), 3)
    elif emotion == "angry":
        cv2.circle(img, (lx, ly), 8, (0, 0, 0, 255), -1)
        cv2.circle(img, (rx, ry), 8, (0, 0, 0, 255), -1)
        # Angry eyebrows \ /
        cv2.line(img, (lx-15, ly-15), (lx+10, ly-5), (0, 0, 0, 255), 4)
        cv2.line(img, (rx+15, ry-15), (rx-10, ry-5), (0, 0, 0, 255), 4)
    elif emotion in ["shocked", "surprised"]:
        cv2.circle(img, (lx, ly), 12, (255, 255, 255, 255), -1)
        cv2.circle(img, (rx, ry), 12, (255, 255, 255, 255), -1)
        cv2.circle(img, (lx, ly), 5, (0, 0, 0, 255), -1)
        cv2.circle(img, (rx, ry), 5, (0, 0, 0, 255), -1)
        # High eyebrows
        cv2.ellipse(img, (lx, ly-20), (15, 10), 0, 180, 360, (0, 0, 0, 255), 2)
        cv2.ellipse(img, (rx, ry-20), (15, 10), 0, 180, 360, (0, 0, 0, 255), 2)
    elif emotion == "confused":
        cv2.circle(img, (lx, ly), 8, (0, 0, 0, 255), -1)
        cv2.circle(img, (rx, ry), 8, (0, 0, 0, 255), -1)
        # Asymmetric eyebrows
        cv2.line(img, (lx-10, ly-15), (lx+10, ly-10), (0, 0, 0, 255), 3)
        cv2.ellipse(img, (rx, ry-20), (15, 10), 0, 180, 360, (0, 0, 0, 255), 3)
    else: # neutral 
        cv2.circle(img, (lx, ly), 8, (0, 0, 0, 255), -1)
        cv2.circle(img, (rx, ry), 8, (0, 0, 0, 255), -1)

    # Female eyelashes
    if gender == 'female' and emotion not in ["happy", "excited", "laugh"]:
        cv2.line(img, (lx+8, ly), (lx+15, ly-8), (0, 0, 0, 255), 2)
        cv2.line(img, (rx-8, ry), (rx-15, ry-8), (0, 0, 0, 255), 2)

    # 4. Mouth
    mouth_y = cy + int(r * 0.3)
    if emotion == "happy":
        cv2.ellipse(img, (cx, mouth_y-10), (25, 20), 0, 0, 180, (0, 0, 0, 255), 4)
    elif emotion in ["excited", "laugh"]:
        # Open happy mouth
        cv2.ellipse(img, (cx, mouth_y-10), (30, 25), 0, 0, 180, (0, 0, 0, 255), -1)
        # Tongue
        cv2.ellipse(img, (cx, mouth_y+5), (15, 10), 0, 0, 180, (50, 50, 255, 255), -1)
    elif emotion == "sad":
        cv2.ellipse(img, (cx, mouth_y+15), (25, 20), 0, 180, 360, (0, 0, 0, 255), 4)
        # A tear
        cv2.circle(img, (lx, ly+25), 5, (255, 200, 0, 255), -1)
    elif emotion in ["shocked", "surprised", "fearful"]:
        cv2.circle(img, (cx, mouth_y+5), 15, (0, 0, 0, 255), -1)
    elif emotion == "angry":
        cv2.ellipse(img, (cx, mouth_y+15), (20, 15), 0, 180, 360, (0, 0, 0, 255), 4)
    elif emotion == "confused":
        cv2.line(img, (cx-15, mouth_y), (cx+15, mouth_y), (0, 0, 0, 255), 3)
    elif emotion == "disgusted":
        cv2.line(img, (cx-20, mouth_y), (cx+20, mouth_y-10), (0, 0, 0, 255), 3)
    else: # neutral
        cv2.line(img, (cx-20, mouth_y), (cx+20, mouth_y), (0, 0, 0, 255), 3)

    return img
