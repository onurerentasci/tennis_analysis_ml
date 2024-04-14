from ultralytics import YOLO
import cv2
import pickle  # Python'da nesnelerin serileştirilmesi ve deserializasyonu için kullanılan bir modüldür

import sys

sys.path.append("../")
from utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # YOLO sınıfından bir nesne oluşturulur
        # model_path parametresi ile model yolu belirtilir

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # court_keypoints -> karedeki kort çizgilerinin koordinatlarını içerir
        # player_detections -> karedeki nesnelerin koordinatlarını içerir
        player_detections_first_frame = player_detections[0]
        # player_detections listesinin ilk elemanı alınır, çünkü kort çizgileri ilk karede tespit edilir
        chosen_player = self.choose_players(
            court_keypoints, player_detections_first_frame
        )
        # choose_players() -> player'ları seçer, kort çizgileri ve player'ların koordinatları kullanılır
        filtered_player_detections = []
        # seçilen player'ların koordinatlarını içerecek liste oluşturulur
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_player
            }
            # player_dict içerisindeki her bir nesne için track_id ve bbox alınır, eğer track_id seçilen player'ların içindeyse bu nesne filtered_player_dict'e eklenir.
            # track_id -> nesnenin takip numarasını içerir
            # bbox -> nesnenin koordinatlarını içerir
            # chosen_player -> seçilen player'ların takip numaralarını içerir
            filtered_player_detections.append(filtered_player_dict)
            # filtered_player_dict filtered_player_detections listesine eklenir.
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        # player'ların kort çizgilerine olan uzaklıklarını içerecek liste oluşturulur, her bir eleman bir tuple'dır
        # tuple'ın ilk elemanı track_id, ikinci eleman ise mesafedir
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            # her player'ın merkezini alır

            min_distance = float("inf")  # sonsuz
            # her bir player'ın kort çizgilerine olan mesafesini hesaplamak için kullanılır.
            # neden sonsuz? -> herhangi bir mesafe sonsuzdan küçük olacaktır. Bu yüzden ilk mesafe sonsuz olarak belirlenir
            for i in range(0, len(court_keypoints), 2):
                # kort çizgilerinin koordinatlarına erişilir
                # ikişer ikişer alınır
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                # i. ve i+1. elemanlar alınır
                distance = measure_distance(player_center, court_keypoint)
                # i. ve i+1. elemanlar arasındaki mesafe hesaplanır
                if distance < min_distance:
                    min_distance = distance
                    # eğer hesaplanan mesafe min_distance'tan küçükse, min_distance güncellenir
            distances.append((track_id, min_distance))
            # distances listesine (track_id, min_distance) tuple'ı eklenir. Bu tuple'ın birinci elemanı track_id, ikinci elemanı ise min_distance'dır

        # distances listesini mesafeye göre sıralar
        distances.sort(key=lambda x: x[1])
        # lambda x: x[1] -> x'in 1. elemanına göre sıralama yapar
        # distances[0][0] ve distances[1][0] -> kort çizgilerine en yakın olan ilk 2 player'ı seçer

        # ilk 2 player'ı seçer
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # read_from_stub -> eğer True ise, tespitlerin kaydedildiği dosyadan okuma yapar
        # stub nedir? -> bir programın belirli bir kısmını temsil eden ve genellikle test amaçlı kullanılan bir program parçasıdır
        # stub_path -> tespitlerin kaydedildiği dosyanın yolu
        player_detections = []

        if read_from_stub and stub_path is not None:
            # read_from_stub ve stub_path değerleri True ise ve stub_path değeri None değilse
            with open(stub_path, "rb") as f:
                # stub_path değerindeki dosya okunur. rb -> read binary
                player_detections = pickle.load(f)
                # pickle.load() -> dosyadaki veriyi okur ve geri döndürür
            return player_detections

        for frame in frames:
            # frames -> video karelerinin listesi
            player_dict = self.detect_frame(frame)
            # detect_frame() -> karedeki nesneleri tespit eder
            player_detections.append(player_dict)

        if stub_path is not None:
            # eğer stub_path değeri None değilse
            with open(stub_path, "wb") as f:
                # stub_path değerindeki dosya yazılır. wb -> write binary
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        # persist parametresi ile takip edilen nesnelerin bir sonraki karede de takip edilmesi sağlanır
        # track metodu ile modelin tahmin yapması sağlanır
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            # box -> tahmin edilen nesnenin bilgilerini içerir
            track_id = int(box.id.tolist()[0])
            # track_id -> tahmin edilen nesnenin takip numarasını içerir
            # id.tolist()[0] -> id değerini alır
            result = box.xyxy.tolist()[0]
            # xyxy.tolist()[0] -> tahmin edilen nesnenin koordinatlarını alır
            # xyxy -> nesnenin koordinatlarını içerir
            object_cls_id = box.cls.tolist()[0]
            # cls.tolist()[0] -> nesnenin sınıfını alır
            object_cls_name = id_name_dict[object_cls_id]
            # id_name_dict[object_cls_id] -> nesnenin sınıf ismini alır
            if object_cls_name == "person":
                player_dict[track_id] = result
                # eğer nesne sınıfı "person" ise player_dict sözlüğüne eklenir
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        # player_dict içerisindeki her bir nesne için bbox çizme işlemi yapılır

        for frame, player_dict in zip(video_frames, player_detections):
            # zip fonksiyonu ile iki farklı listeyi birleştirir
            for track_id, bbox in player_dict.items():
                # player_dict içerisindeki her bir nesne için bbox çizme işlemi yapılır
                x1, y1, x2, y2 = bbox
                # bbox içerisindeki koordinatlar alınır
                cv2.putText(  # cv2.putText() -> kare üzerine metin yazma işlemi yapar
                    frame,  # frame -> metnin yazılacağı kare
                    f"Player {track_id}",  # metin
                    (int(bbox[0]), int(bbox[1] - 10)),  # metnin konumu
                    cv2.FONT_HERSHEY_SIMPLEX,  # metnin fontu
                    0.9,  # metnin boyutu
                    (0, 255, 0),  # metnin rengi
                    2,  # metnin kalınlığı
                )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,  # cv2.rectangle() -> kare üzerine dikdörtgen çizme işlemi yapar
                    # (int(x1), int(y1)), (int(x2), int(y2)) -> dikdörtgenin koordinatları
                )
            output_video_frames.append(frame)
            # çizilen kare output_video_frames listesine eklenir
        return output_video_frames
