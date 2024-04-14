# Tennis Analysis: AI ile Saha Hakimiyeti

Bu proje, yapay zekanın (AI) tenis maçlarını analiz etmek için nasıl kullanılabileceğini gösteren bir örnektir. Proje, bir tenis kortunun keypointlerini (köşe noktaları, çizgiler vb.) ve sahadaki oyuncuların ve topun konumlarını belirlemek için bir derin öğrenme modeli kullanır. Bu bilgiler daha sonra ekranda bir minimap üzerinde görselleştirilir, bu da size sahadaki durum hakkında anlık bir bakış açısı sunar.

**Proje Nasıl Çalışır:**

**1. Derin Öğrenme Modeli:**

- Proje, **ResNet50** adı verilen bir derin öğrenme modeli kullanır. Bu model, ImageNet veri kümesi üzerinde eğitilmiştir ve 14 anahtar noktanın (köşeler, çizgiler vb.) konumunu tahmin etmek için kullanılabilir.
- **torch** kütüphanesi modelin yükleme ve çalıştırma işlemleri için kullanılır.
- **torchvision.models** kütüphanesi ResNet50 modelini oluşturmak için kullanılır.

**2. Görüntü İşleme:**

- Bir video dosyasından gelen görüntüler, önceden işlenir ve modele beslenir. Bu işlem, görüntü boyutunun küçültülmesini, renk uzayının değiştirilmesini ve normalleştirmeyi içerir.
- **cv2** kütüphanesi görüntü işleme işlemleri için kullanılır.

**3. Keypoint Tahmini:**

- Model, her görüntüde 14 anahtar noktanın konumunu tahmin eder. Bu tahminler, piksel koordinatları olarak ifade edilir.
- **torch.no_grad()** fonksiyonu modeli gradyan hesaplaması yapmadan ileri geçiş için hazırlar.

**4. Minimap Görselleştirmesi:**

- Tahmin edilen anahtar noktalar, ekranda bir minimap üzerine çizilir. Bu minimap, sahanın genel bir görünüşünü ve oyuncuların ve topun konumlarını gösterir.
- **torchvision** kütüphanesi görüntüyü dönüştürmek ve tensöre dönüştürmek için kullanılır.

**Kullanım Örneği:**

Proje demo videosu, bir tenis maçının analizini gösterir. Videoda, modelin keypointleri doğru bir şekilde tahmin ettiği ve minimap'in oyunu gerçek zamanlı olarak takip ettiği görülebilir.

[![Video Örneği](https://img.youtube.com/vi/CgR_yhtCZME/0.jpg)](https://www.youtube.com/watch?v=CgR_yhtCZME)

**Model Dosyaları**

Projeyi çalıştırmak için gerekli olan model dosyalarına aşağıdaki Google Drive bağlantısından erişebilirsiniz:

[Model Dosyaları - Google Drive](https://drive.google.com/drive/folders/1n8mLQ08vsNT33QrZyv7a__aTXRk7Imnx?usp=sharing)

**Proje Kapsamı:**

Bu proje, yapay zekanın tenis maçlarını analiz etmek için nasıl kullanılabileceğini gösteren bir örnektir. Gerçek bir tenis maçı analiz uygulamasında kullanılmadan önce daha fazla geliştirilmesi gerekir.
