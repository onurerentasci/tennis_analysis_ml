from ultralytics import YOLO
import cv2
import pickle  # Python'da nesnelerin serileştirilmesi ve deserializasyonu için kullanılan bir modüldür
import pandas as pd


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # YOLO sınıfından bir nesne oluşturulur
        # model_path parametresi ile model yolu belirtilir
        # burada init metodu ile model yolu belirtilir

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # ball_positions listesindeki her bir nesne için 1. eleman alınır
        # neden 1? -> 1. eleman, nesnenin koordinatlarını içerir
        # x.get(1, []) -> x'in 1. elemanını alır, eğer x'in 1. elemanı yoksa boş liste döndürür

        # listeyi pandas DataFrame'e dönüştürme işlemi yapılır
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # eksik verileri doldurma işlemi yapılır
        df_ball_positions = df_ball_positions.interpolate(method="linear")
        # neden linear? -> eksik verileri iki bilinen veri arasında bir doğru çizerek doldurur
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        # df_ball_positions verisini numpy dizisine dönüştürür ve listeye çevirir
        # [{1:x} for x in df_ball_positions.to_numpy().tolist()] -> df_ball_positions verisini numpy dizisine dönüştürür ve listeye çevirir,
        # her bir elemanın başına 1 ekler. Çünkü her bir elemanın bir track_id'si vardır

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        df_ball_positions["ball_hit"] = 0

        df_ball_positions["mid_y"] = (
            df_ball_positions["y1"] + df_ball_positions["y2"]
        ) / 2
        df_ball_positions["mid_y_rolling_mean"] = (
            df_ball_positions["mid_y"]
            .rolling(window=5, min_periods=1, center=False)
            .mean()
        )
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()
        minimum_change_frames_for_hit = 25
        for i in range(
            1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)
        ):
            negative_position_change = (
                df_ball_positions["delta_y"].iloc[i] > 0
                and df_ball_positions["delta_y"].iloc[i + 1] < 0
            )
            positive_position_change = (
                df_ball_positions["delta_y"].iloc[i] < 0
                and df_ball_positions["delta_y"].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(
                    i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1
                ):
                    negative_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] > 0
                        and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                        df_ball_positions["delta_y"].iloc[i] < 0
                        and df_ball_positions["delta_y"].iloc[change_frame] > 0
                    )

                    if (
                        negative_position_change
                        and negative_position_change_following_frame
                    ):
                        change_count += 1
                    elif (
                        positive_position_change
                        and positive_position_change_following_frame
                    ):
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, "ball_hit"] = 1

        frame_nums_with_ball_hits = df_ball_positions[
            df_ball_positions["ball_hit"] == 1
        ].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # frames -> video karelerinin listesi
        # read_from_stub -> eğer True ise, tespitlerin kaydedildiği dosyadan okuma yapar
        # stub_path -> tespitlerin kaydedildiği dosyanın yolu
        ball_detections = []

        if read_from_stub and stub_path is not None:
            # read_from_stub ve stub_path değerleri True ise ve stub_path değeri None değilse
            with open(stub_path, "rb") as f:
                # stub_path değerindeki dosya okunur. rb -> read binary
                ball_detections = pickle.load(f)
                # pickle.load() -> dosyadaki veriyi okur ve geri döndürür
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                # stub_path değerindeki dosya yazılır. wb -> write binary
                pickle.dump(ball_detections, f)
                # pickle.dump() -> veriyi dosyaya yazar

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        # model.predict() -> modeli kullanarak tahmin yapar
        # conf -> güven eşiği
        # [0] -> tahminlerin ilk elemanı alınır

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
            # box.xyxy -> tahminlerin koordinatları

        return ball_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        # ball_dict içerisindeki her bir nesne için bbox çizme işlemi yapılır

        for frame, ball_dict in zip(video_frames, player_detections):
            # zip fonksiyonu ile iki farklı listeyi birleştirir
            for track_id, bbox in ball_dict.items():
                # ball_dict içerisindeki her bir nesne için bbox çizme işlemi yapılır
                x1, y1, x2, y2 = bbox
                # bbox içerisindeki koordinatlar alınır
                cv2.putText(  # cv2.putText() -> kare üzerine metin yazma işlemi yapar
                    frame,  # frame -> metnin yazılacağı kare
                    f"Ball {track_id}",  # metin
                    (int(bbox[0]), int(bbox[1] - 10)),  # metnin konumu
                    cv2.FONT_HERSHEY_SIMPLEX,  # metnin fontu
                    0.9,  # metnin boyutu
                    (255, 0, 255),  # metnin rengi
                    2,  # metnin kalınlığı
                )

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 255),
                    2,  # cv2.rectangle() -> kare üzerine dikdörtgen çizme işlemi yapar
                    # (int(x1), int(y1)), (int(x2), int(y2)) -> dikdörtgenin koordinatları
                )
            output_video_frames.append(frame)
            # çizilen kare output_video_frames listesine eklenir
        return output_video_frames
