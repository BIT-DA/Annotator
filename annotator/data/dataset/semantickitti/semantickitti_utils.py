LABEL_NAME_MAPPING = {
    0: 'unlabeled',  # 0
    1: 'outlier',  # 0
    10: 'car',  # 1
    11: 'bicycle',  # 2
    13: 'bus',  # 5
    15: 'motorcycle',  # 3
    16: 'on-rails',  # 0
    18: 'truck',  # 4
    20: 'other-vehicle',  # 0
    30: 'person',  # 6
    31: 'bicyclist',  # 0
    32: 'motorcyclist',  # 0
    40: 'road',  # 7
    44: 'parking',  # 0
    48: 'sidewalk',  # 8
    49: 'other-ground',  # 9
    50: 'building',  # 12
    51: 'fence',  # 12
    52: 'other-structure',  # 0
    60: 'lane-marking',  # 7
    70: 'vegetation',  # 10
    71: 'trunk',  # 0
    72: 'terrain',  # 11
    80: 'pole',  # 12
    81: 'traffic-sign',  # 12
    99: 'other-object',  # 0
    252: 'moving-car',  # 1
    253: 'moving-bicyclist',  # 0
    254: 'moving-person',  # 6
    255: 'moving-motorcyclist',  # 0
    256: 'moving-on-rails',  # 0
    257: 'moving-bus',  # 5
    258: 'moving-truck',  # 4
    259: 'moving-other-vehicle'  # 0
}

color_map = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}

LEARNING_MAP_19 = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

LEARNING_MAP_12 = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 0,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 0,  # "other-vehicle"
    30: 6,  # "person"
    31: 0,  # "bicyclist"
    32: 0,  # "motorcyclist"
    40: 7,  # "road"
    44: 0,  # "parking"
    48: 8,  # "sidewalk"
    49: 9,  # "other-ground"
    50: 12,  # "building"
    51: 12,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 7,  # "lane-marking" to "road" ---------------------------------mapped
    70: 10,  # "vegetation"
    71: 0,  # "trunk"
    72: 11,  # "terrain"
    80: 12,  # "pole"
    81: 12,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 0,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 0,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 0,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 0,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

LEARNING_MAP_7 = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 0,  # "bicycle"
    13: 0,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 0,  # "motorcycle"
    16: 0,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 0,  # "truck"
    20: 0,  # "other-vehicle"
    30: 2,  # "person"
    31: 0,  # "bicyclist"
    32: 0,  # "motorcyclist"
    40: 3,  # "road"
    44: 3,  # "parking"
    48: 4,  # "sidewalk"
    49: 0,  # "other-ground"
    50: 6,  # "building"
    51: 6,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 3,  # "lane-marking" to "road" ---------------------------------mapped
    70: 7,  # "vegetation"
    71: 7,  # "trunk"
    72: 5,  # "terrain"
    80: 6,  # "pole"
    81: 6,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 0,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 2,  # "moving-person" to "person" ------------------------------mapped
    255: 0,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 0,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 0,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 0,  # "moving-truck" to "truck" --------------------------------mapped
    259: 0,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

LEARNING_MAP_11 = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 0,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 0,  # "motorcycle"
    16: 0,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 0,  # "truck"
    20: 0,  # "other-vehicle"
    30: 3,  # "person"
    31: 4,  # "bicyclist"
    32: 4,  # "motorcyclist"
    40: 5,  # "road"
    44: 0,  # "parking"
    48: 0,  # "sidewalk"
    49: 0,  # "other-ground"
    50: 6,  # "building"
    51: 7,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 5,  # "lane-marking" to "road" ---------------------------------mapped
    70: 8,  # "vegetation"
    71: 9,  # "trunk"
    72: 0,  # "terrain"
    80: 10,  # "pole"
    81: 11,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 4,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 3,  # "moving-person" to "person" ------------------------------mapped
    255: 4,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 0,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 0,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 0,  # "moving-truck" to "truck" --------------------------------mapped
    259: 0,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

LEARNING_MAP_INV = {  # inverse of previous map
    0: 0,  # "unlabeled", and others ignored
    1: 10,  # "car"
    2: 11,  # "bicycle"
    3: 15,  # "motorcycle"
    4: 18,  # "truck"
    5: 20,  # "other-vehicle"
    6: 30,  # "person"
    7: 31,  # "bicyclist"
    8: 32,  # "motorcyclist"
    9: 40,  # "road"
    10: 44,  # "parking"
    11: 48,  # "sidewalk"
    12: 49,  # "other-ground"
    13: 50,  # "building"
    14: 51,  # "fence"
    15: 70,  # "vegetation"
    16: 71,  # "trunk"
    17: 72,  # "terrain"
    18: 80,  # "pole"
    19: 81,  # "traffic-sign"
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
