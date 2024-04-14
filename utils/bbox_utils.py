def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    # iki nokta arasındaki mesafeyi hesaplar
    # bu mesafe bir üçgenin hipotenüsüne eşittir, bu yüzden hipotenüs formülü kullanılır

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float("inf")
    keypoint_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index * 2], keypoints[keypoint_index * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            keypoint_ind = keypoint_index

    return keypoint_ind

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]
    # bbox'un yüksekliğini hesaplar
    # bbox[3]? -> y2, cünkü bbox'un 4 koordinatı var, x1, y1, x2, y2

def mesaure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
    # p1[0]? -> x1, çünkü p1'in 2 koordinatı var, x1, y1
    # abs(p1[0] - p2[0]) -> x1 ve x2 arasındaki mesafeyi hesaplar.

def get_center_of_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
