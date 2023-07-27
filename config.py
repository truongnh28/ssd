cfgs = {
    "num_class": 21,  # VOC data include 20 class + 1 background class
    "input_size": 300,  # SSD300
    "bbox_aspect_num": (4, 6, 6, 6, 4, 4),  # tỷ lệ khung hình từ source_1 -> source_6
    "feature_map": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "min_size": [30, 60, 111, 162, 213, 264],
    "max_size": [60, 111, 162, 213, 264, 315],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}
