from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

import cv2


def main():
    # video okuma işlemi
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # player tespit işlemi
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")

    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )  # player tespitlerini alır, eğer stub_path değeri None değilse, tespitlerin kaydedildiği dosyadan okuma yapar
    # tespitlerin kaydedildiği dosyanın yolu tracker_stubs/player_detections.pkl'dir
    # .pkl nedir? -> Python'da pickle modülü ile oluşturulan dosyaların uzantısıdır

    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections.pkl",
    )

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    # topun pozisyonlarını lineer olarak ardışık kareler arasındaki topun pozisyonunu tahmin eder

    # Kort Çizgileri Tespiti

    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    # court_keypoints -> karedeki kort çizgilerinin koordinatlarını içerir
    # predict metodu ile karedeki kort çizgileri tespit edilir
    # video_frames[0] -> video karelerinin ilki

    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    # mini kort çizme işlemi
    mini_court = MiniCourt(video_frames[0])

    # top vuruşlarını tespit etme işlemi
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # mini kort pozisyonlarını dönüştürme işlemi
    player_mini_court_detection, ball_mini_court_detection = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
    )

    # Draw output

    ## player BBox çizme işlemi
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Kort çizgileri çizme işlemi
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )

    # mini kort çizme işlemi
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detection
    )
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detection, color=(255, 0, 255)
    )

    # player'ların kort çizgilerine olan uzaklıklarını hesaplama işlemi
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    # frame numaralarını ekranın sol üst köşesine yazdırma işlemi
    for i, frame in enumerate(output_video_frames):
        frame = cv2.putText(
            frame,
            f"Frame: {str(i)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
