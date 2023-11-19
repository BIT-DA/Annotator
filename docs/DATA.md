# Data Preparation

### Overall Structure

```
└── data_root 
    └── Synlidar
    └── SemanticKITTI
    └── SemanticPOSS
    └── nuScenes
```



### SynLiDAR
To install the [SynLiDAR](https://github.com/xiaoaoran/SynLiDAR) dataset, download the data and annotations. Unpack the compressed file(s) into `./data_root/Synlidar` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── Synlidar  
    └── sequences
        ├── 00 
        │    └── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── labels <- contains the .label files; a .label file contains the labels of the points in a point cloud
        ├── 01 
        ├── ...  
        ├── 11 
        └── 12
```


### SemanticKITTI

To install the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset. Unpack the compressed file(s) into `./data_root/SemanticKITTI` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── SemanticKITTI
    └── sequences
        ├── 00 
        │    └── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── labels <- contains the .label files; a .label file contains the labels of the points in a point cloud
        ├── 01 
        ├── ...  
        ├── 20 
        └── 21
```

### SemanticPOSS
To install the [SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html) dataset, download the data and annotations. Unpack the compressed file(s) into `./data_root/SemanticPOSS` and re-organize the data structure. Your folder structure should end up looking like this:


```
└── SemanticPOSS
    └── sequences
        ├── 00 
        │    └── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── labels <- contains the .label files; a .label file contains the labels of the points in a point cloud
        ├── 01 
        ├── 02
        ├── 03
        ├── 04
        └── 05
```

### nuScenes

To install the [nuScenes-lidarseg](https://www.nuscenes.org/nuscenes) dataset, download the data, annotations, and other files from https://www.nuscenes.org/download. Unpack the compressed file(s) into `./data_root/nuScenes` and your folder structure should end up looking like this:

```
└── nuScenes  
    ├── Usual nuscenes folders (i.e. samples, sweep)
    │
    ├── lidarseg
    │   └── v1.0-{mini, test, trainval} <- contains the .bin files; a .bin file 
    │                                      contains the labels of the points in a 
    │                                      point cloud (note that v1.0-test does not 
    │                                      have any .bin files associated with it)
    │
    └── v1.0-{mini, test, trainval}
        ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
        ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
        └── category.json  <- contains the categories of the labels (note that the 
                              category.json from nuScenes v1.0 is overwritten)
```


### References

Please consider site the original papers of the datasets if you find them helpful to your research.
