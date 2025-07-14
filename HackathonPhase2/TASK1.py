import cv2
import pandas as pd
dataset = []
def click_event(event, x, y, flags, param):
    global dataset
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Koordinatlar: ({x}, {y})")
        pattern_type = input("Desen türü (Circle/Cross-sign): ").strip()
        if pattern_type in ["Circle", "Cross-sign"]:
            dataset.append((x, y, pattern_type))
            print(f"Eklendi: {x}, {y}, {pattern_type}")
        else:
            print("Geçersiz desen türü!")
cap = cv2.VideoCapture(0)
ret, img = cap.read()
cap.release()
if ret:
    cv2.imshow("Yakalanan Görüntü", img)
    cv2.setMouseCallback("Yakalanan Görüntü", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    df = pd.DataFrame(dataset, columns=["x", "y", "Pattern Type"])
    df.to_csv("dataset.csv", index=False)
    print("Veri kümesi kaydedildi!")
else:
    print("Görüntü yakalanamadı.")
