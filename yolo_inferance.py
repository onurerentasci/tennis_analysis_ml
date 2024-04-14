from ultralytics import YOLO

# Object Detection

# yoloV8x modeli tercih ediliyor
# doğrudan bir path ifadesi yazılırsa dosya yolundaki model ile run yapılabilir
# model = YOLO("models/yolo5_last.pt")

# result = model.predict(source="input_videos/input_video.mp4", conf=0.2, save=True)

# conf ifadesi, modelin tahmin yaparken kullanacağı güven eşiğini belirtir.
# Bu durumda, conf=0.2 demek, modelin yalnızca en az %20 güvenle tahmin ettiği sonuçları dikkate alacağı anlamına gelir.

# eğer kendi hazırladığımız model yerine yolonun kendi modellerini kullanmak istersek,
# bu işlemden sonra hazır olan Yolov8x modeli indirilecek ve source üzerinde çalışacak
# runs klasörü otomatik oluşacak ve tespit edilen görselin çıktısı burada saklanacak

# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)

# -------------------------------------------------------------------------------------------------

# Object Tracking

model = YOLO(model="yolov8x")
result = model.track(source="input_videos/input_video.mp4", conf=0.2, save=True)

