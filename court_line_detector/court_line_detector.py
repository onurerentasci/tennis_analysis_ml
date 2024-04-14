import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # resnet50 modeli oluşturulur, ağırlıklar varsayılan olarak yüklenir, yani ImageNet veri kümesinde eğitilmiş ağırlıklar kullanılır
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        # modelin çıktı katmanı değiştirilir, çünkü modelin çıktı katmanı ImageNet veri kümesi için eğitilmiştir
        # neden Linear? -> modelin çıktı katmanı tam bağlıdır, yani her bir nöron bir önceki katmandaki tüm nöronlarla bağlıdır
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        # modelin ağırlıkları yüklenir
        # map_location="cpu" -> modelin ağırlıklarının CPU üzerinde yüklenmesini sağlar

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # Görüntü 224x224 piksel olarak yeniden boyutlandırılır
                transforms.Resize((224, 224)),
                # PIL görüntüsü PyTorch tensörüne dönüştürülür.
                # bu dönüşüm piksel değerlerini [0,1] aralığında ölçeklendirir
                transforms.ToTensor(),
                # Bu dönüşüm, normalleştirme işlemi yapar.
                # Her bir kanalı, belirtilen ortalama (mean) ve standart sapma (std) değerlerine göre normalleştirir.
                # Bu, genellikle eğitim sürecinde modelin daha hızlı ve daha kararlı öğrenmesine yardımcı olmak için yapılır.
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)
        # unsqueeze(0) -> tensörün boyutunu artırır, tensörün boyutunu (1, C, H, W) yapar
        # 1 -> batch boyutu, C -> kanal sayısı, H -> yükseklik, W -> genişlik

        with torch.no_grad():
            # gradyan hesaplaması yapmadan ileri doğru geçiş yapar
            output = self.model(image_tensor)
            # modeli kullanarak ileri doğru geçiş yapar

        keypoints = output.squeeze().cpu().numpy()
        # çıktı tensörünün boyutunu azaltır, tensörün boyutunu (C, H, W) yapar
        original_h, original_w = image.shape[:2]
        # orijinal görüntünün yüksekliği ve genişliği alınır

        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        # x ve y koordinatları yeniden boyutlandırılır
        # neden 224? -> modelin giriş boyutu 224x224 piksel olduğu için

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            # keypoints listesindeki her bir eleman için x ve y koordinatları alınır
            # 2'şer 2'şer artırarak x ve y koordinatları alınır
            # Neden? -> çünkü x ve y koordinatları sırayla gelir
            
            x, y = keypoints[i], keypoints[i + 1]
            # x ve y koordinatları alınır
            
            cv2.putText(
                image,
                str(i // 2),
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
