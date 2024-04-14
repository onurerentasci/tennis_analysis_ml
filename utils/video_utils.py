import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        # cap.read() videoda bir sonraki kareyi okur.
        # ret bir bool değerdir karenin başında okunup okunmadığını belirtir
        # frame -> okunan görüntü karesi

        if not ret:
            # eğer ret = false dönerse video okuma işlemi sonlanır
            break
        frames.append(frame)
        # eğer kare başarıyla okunduysa frames listesine eklenşr
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # XVID -> video codec
    # XVID yerine MP4V, X264, X265 gibi codec'ler de kullanılabilir.
    # video codec belirlenir.
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    # video yazmak için VideoWriter nesnesi oluşturulur
    # output_video_path -> video kaydedilecek yol
    # fourcc -> video codec
    # frameSize -> video karesinin boyutu

    for frame in output_video_frames:
        out.write(frame)
        # bu döngü ile her bir kareyi video dosyasına yazar
    out.release()
