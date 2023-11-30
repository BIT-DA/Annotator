LABEL_NAME_MAPPING = {
    0: "unlabeled",  # 0
    1: "car",  # 1
    2: "pick-up",  # 0
    3: "truck",  # 4
    4: "bus",  # 5
    5: "bicycle",  # 2
    6: "motorcycle",  # 3
    7: "other-vehicle",  # 0
    8: "road",  # 7
    9: "sidewalk",  # 8
    10: "parking",  # 0
    11: "other-ground",  # 9
    12: "female",  # 6
    13: "male",  # 6
    14: "kid",  # 6
    15: "crowd",  # 0 # multiple person that are very close
    16: "bicyclist",  # 0
    17: "motorcyclist",  # 0
    18: "building",  # 12
    19: "other-structure",  # 0
    20: "vegetation",  # 10
    21: "trunk",  # 0
    22: "terrain",  # 11
    23: "traffic-sign",  # 12
    24: "pole",  # 12
    25: "traffic-cone",  # 13
    26: "fence",  # 12
    27: "garbage-can",  # 0
    28: "electric-box",  # 0
    29: "table",  # 0
    30: "chair",  # 0
    31: "bench",  # 0
    32: "other-object",  # 0
}

LEARNING_MAP_KITTI = {
    0: 0,  # "unlabeled"
    1: 1,  # "car"
    2: 0,  # "pick-up"
    3: 4,  # "truck"
    4: 5,  # "bus"
    5: 2,  # "bicycle"
    6: 3,  # "motorcycle"
    7: 5,  # "other-vehicle"
    8: 9,  # "road"
    9: 11,  # "sidewalk"
    10: 10,  # "parking"
    11: 12,  # "other-ground"
    12: 6,  # "female"
    13: 6,  # "male"
    14: 6,  # "kid"
    15: 0,  # "crowd"
    16: 7,  # "bicyclist"
    17: 8,  # "motorcyclist"
    18: 13,  # "building"
    19: 0,  # "other-structure"
    20: 15,  # "vegetation"
    21: 16,  # "trunk"
    22: 17,  # "terrain"
    23: 19,  # "traffic-sign"
    24: 18,  # "pole"
    25: 0,  # "traffic-cone"
    26: 14,  # "fence"
    27: 0,  # "garbage-can"
    28: 0,  # "electric-box"
    29: 0,  # "table"
    30: 0,  # "chair"
    31: 0,  # "bench"
    32: 0,  # "other-object"
}

LEARNING_MAP_nuscenes = {
    0: 0,  # "unlabeled"
    1: 1,  # "car"
    2: 0,  # "pick-up"
    3: 4,  # "truck"
    4: 5,  # "bus"
    5: 2,  # "bicycle"
    6: 3,  # "motorcycle"
    7: 0,  # "other-vehicle"
    8: 7,  # "road"
    9: 8,  # "sidewalk"
    10: 0,  # "parking"
    11: 9,  # "other-ground"
    12: 6,  # "female"
    13: 6,  # "male"
    14: 6,  # "kid"
    15: 0,  # "crowd"
    16: 0,  # "bicyclist"
    17: 0,  # "motorcyclist"
    18: 12,  # "building"
    19: 0,  # "other-structure"
    20: 10,  # "vegetation"
    21: 0,  # "trunk"
    22: 11,  # "terrain"
    23: 12,  # "traffic-sign"
    24: 12,  # "pole"
    25: 13,  # "traffic-cone"
    26: 12,  # "fence"
    27: 0,  # "garbage-can"
    28: 0,  # "electric-box"
    29: 0,  # "table"
    30: 0,  # "chair"
    31: 0,  # "bench"
    32: 0,  # "other-object"
}

LEARNING_MAP_7 = {
    0: 0,  # "unlabeled"
    1: 1,  # "car"
    2: 1,  # "pick-up"
    3: 0,  # "truck"
    4: 0,  # "bus"
    5: 0,  # "bicycle"
    6: 0,  # "motorcycle"
    7: 0,  # "other-vehicle"
    8: 3,  # "road"
    9: 4,  # "sidewalk"
    10: 3,  # "parking"
    11: 0,  # "other-ground"
    12: 2,  # "female"
    13: 2,  # "male"
    14: 2,  # "kid"
    15: 2,  # "crowd"
    16: 0,  # "bicyclist"
    17: 0,  # "motorcyclist"
    18: 6,  # "building"
    19: 0,  # "other-structure"
    20: 7,  # "vegetation"
    21: 7,  # "trunk"
    22: 5,  # "terrain"
    23: 6,  # "traffic-sign"
    24: 6,  # "pole"
    25: 0,  # "traffic-cone"
    26: 6,  # "fence"
    27: 0,  # "garbage-can"
    28: 0,  # "electric-box"
    29: 0,  # "table"
    30: 0,  # "chair"
    31: 0,  # "bench"
    32: 0,  # "other-object"
}

LEARNING_MAP_13 = {
    0: 0,  # "unlabeled"
    1: 1,  # "car"
    2: 0,  # "pick-up"
    3: 0,  # "truck"
    4: 0,  # "bus"
    5: 2,  # "bicycle"
    6: 0,  # "motorcycle"
    7: 0,  # "other-vehicle"
    8: 5,  # "road"
    9: 0,  # "sidewalk"
    10: 0,  # "parking"
    11: 0,  # "other-ground"
    12: 3,  # "female"
    13: 3,  # "male"
    14: 3,  # "kid"
    15: 0,  # "crowd"
    16: 4,  # "bicyclist"
    17: 4,  # "motorcyclist"
    18: 6,  # "building"
    19: 0,  # "other-structure"
    20: 8,  # "vegetation"
    21: 9,  # "trunk"
    22: 0,  # "terrain"
    23: 11,  # "traffic-sign"
    24: 10,  # "pole"
    25: 13,  # "traffic-cone"
    26: 7,  # "fence"
    27: 12,  # "garbage-can"
    28: 0,  # "electric-box"
    29: 0,  # "table"
    30: 0,  # "chair"
    31: 0,  # "bench"
    32: 0,  # "other-object"
}

LEARNING_IGNORE = {  # Ignore classes
    0: True,  # "unlabeled", and others ignored
    1: False,  # "car"
    2: False,  # "bicycle"
    3: False,  # "motorcycle"
    4: False,  # "truck"
    5: False,  # "other-vehicle"
    6: False,  # "person"
    7: False,  # "bicyclist"
    8: False,  # "motorcyclist"
    9: False,  # "road"
    10: False,  # "parking"
    11: False,  # "sidewalk"
    12: False,  # "other-ground"
    13: False,  # "building"
    14: False,  # "fence"
    15: False,  # "vegetation"
    16: False,  # "trunk"
    17: False,  # "terrain"
    18: False,  # "pole"
    19: False,  # "traffic-sign"
}
