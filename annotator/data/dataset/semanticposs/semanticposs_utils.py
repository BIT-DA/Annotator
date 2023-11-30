LABEL_NAME_MAPPING = {
    0: "unlabeled",
    4: "1 person",
    5: "2+ person",
    6: "rider",
    7: "car",
    8: "trunk",
    9: "plants",
    10: "traffic sign 1",  # standing sign
    11: "traffic sign 2",  # hanging sign
    12: "traffic sign 3",  # high/big hanging sign
    13: "pole",
    14: "garbage-can",
    15: "building",
    16: "cone/stone",
    17: "fence",
    21: "bike",
    22: "ground",
}

color_map = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [245, 150, 100],
    3: [245, 230, 100],
    4: [250, 80, 100],
    5: [150, 60, 30],
    6: [255, 0, 0],
    7: [180, 30, 80],
    8: [255, 0, 0],
    9: [30, 30, 255],
    10: [200, 40, 255],
    11: [90, 30, 150],
    12: [255, 0, 255],
    13: [255, 150, 255],
    14: [75, 0, 75],
    15: [75, 0, 175],
    16: [0, 200, 255],
    17: [50, 120, 255],
    18: [0, 150, 255],
    19: [170, 255, 150],
    20: [0, 175, 0],
    21: [0, 60, 135],
    22: [80, 240, 150],
}

LEARNING_MAP_13 = {
    0: 0,  # "unlabeled",
    1: 0,
    2: 0,
    3: 0,
    4: 3,  # "person",
    5: 3,  # "person",
    6: 4,  # "rider",
    7: 1,  # "car",
    8: 9,  # "trunk",
    9: 8,  # "plants",
    10: 11,  # "traffic sign"
    11: 11,  # "traffic sign"
    12: 11,  # "traffic sign"
    13: 10,  # "pole",
    14: 12,  # "garbage-can",
    15: 6,  # "building",
    16: 13,  # "cone/stone",
    17: 7,  # "fence",
    18: 0,
    19: 0,
    20: 0,
    21: 2,  # "bike",
    22: 5,  # "ground"
}