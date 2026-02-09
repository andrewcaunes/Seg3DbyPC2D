"""
Script made by Andrew Caunes.
example use: 
python3 -m seg_3D_by_2D.core.classes_dicts 
    --cs1 uda 
    --cs2 nuscenes_eval
This script defines various class systems (for any context, 2D or 3D) and conversion functions between them.

The global_ext class system is the reference class system used for converting all other systems.
Defining a class system : 
    - Define the classes by a tuple of strings.
    (tip for next step : run classes_dicts.py --classes_system_1 [desired class system] --classes_system_2 global_ext to see the unmapped classes)
    - If necessary, define a disembiguation dict to convert to global_ext classes.
        -> For each class in the system, map it to the most relevant global_ext class.
        (e.g. different naming "road" -> "drivable_surface" or missing classes. Consider adding the class to global_ext if it is missing)
    - Define a disembiguation dict to convert from global_ext classes.
        -> For each class in the system, go through global classes and map the relevant ones. Dont map the irrelevant ones 
        as they whill automatically be mapped to the backgroud_class.
/!\ Changing the global_ext classes implies changing all other class systems, mostly add the new class to the "from_global" disembiguation dict.

Possible Tasks :
 - Convert from system-1 to system-2 :
     (- Convert system-1 inds to str)
     - Convert system-1 to global
     - Convert global to system-2
     (- Convert system-2 to inds)
    example : initial->nuscenes
        > labels_str = initial_classes_system.to_cls(labels)
        > labels_str = initial_classes_system.to_global(labels_str)
        > labels_str = nuscenes_classes_system.from_global(labels_str)
        > labels = nuscenes_classes_system.to_ind(labels_str)
 - Compare system-1 and system-2 results
     - Decide which system to use for comparison. Global is not always the best choice since it might be too detailed and some classes should be merged.
     - Convert both system-1 and system-2 to the chosen system (can be one of the two or a third one)
     
Guidelines for setting up a new Source-to-Target experiment:
    1. Define the class system S for the source dataset (as described above)
    2. Define the class system T for the target dataset
    3. Define the class system T' for the classes system chosen for evaluation
    4. Train model on dense_GT annotations generated from S using the T' classes system (e.g. set args.dataset_name=S and args.classes=T')
    5. Eval model on dense_GT annotations generated from T using the T' classes system (e.g. set args.dataset_name=T and args.classes=T')
"""


import os
import shutil
import argparse
import logging
logging.basicConfig(format='[%(module)s | l.%(lineno)d] %(message)s')
logging.getLogger().setLevel(logging.INFO)
import numpy as np


def invert_dict(d):
    return {v: k for k, v in d.items()}

dataset_classes_systems = (
    # ORIGINAL DATASET CLASSES SYSTEMS
    "nuscenes_uda",
    "nuscenes_dglss",
    "nuscenes_sk19",
    "semantickitti_uda",
    "semantickitti_dglss",
    "semantickitti_sk19",
    "semantickitti_ns16",
    "waymo",
    "synlidar",
    "synlidar_dglss",
    "synlidar_sk19",
    # OLDER SYSTEMS, MIGHT BE DEPRECARED
    "nuscenes_eval",
    "mapillary_nbs",
    "mapillary_ext",
)
pred_classes_systems = (
    "uda",
    "dglss", # from https://ieeexplore.ieee.org/document/10203512/
    "sk19",
    "ns16",
    "rare",
    "global_ext"
)
# combine all
systems = dataset_classes_systems + pred_classes_systems

## reference global classes
# This is the reference global class system used for converting all other systems.
# It is built to be broad enough to contain all future classes.
global_ext_classes=(
    "background", # general garbage class for all classes that are not in the system and outliers
    "drivable", # drivable is more general than road. Use road when possible.
        "road", 
        "parking",
        "road_marking",
    "other_ground",
    "sidewalk",
    "terrain",
    "sky",
    "manmade", # manmade is more general than all manmade classes, use the specific class when possible.
        "building",
        "traffic_sign",
        "other_sign",
        "traffic_light",
        "pothole",
        "manhole",
        "street_light",
        "pole",
        "traffic_cone",
        "wall",
            "barrier",
            "fence",
    "vegetation", # vegetation is more general than trunk. Use trunk when possible.
        "trunk",
    "vehicle", # vehicle is more general than all vehicle classes. Use the specific class when possible.
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "emergency_vehicle",
        "bus",
        "other_vehicle",
        "construction_vehicle",
        "ego_vehicle",
        "on_rails",
    "pedestrian",
        "bicyclist",
        "motorcyclist",
        "stroller",
        "wheelchair",
        "personal_mobility",
    "animal",
)



color_palette_float = {0:  (np.array([0,0,0])), # color, is_instance_class
                1:  (np.array([0,1,0])/1.4),
                2:  (np.array([0,1,1])/1.4),
                3:  (np.array([1,0,0])/1.4),
                4:  (np.array([1,0,1])/1.4),
                5:  (np.array([1,1,0])/1.4),
                6:  (np.array([0.8,1.1,1.1])/1.2),
                7:  (np.array([0.8,0,0.8])/1.2),
                8:  (np.array([0.3,0.8,0])/1.2),
                9:  (np.array([0.8,0.5,0])/1.2),
                10: (np.array([0.8,0,0])/1.2),
                11: (np.array([0.8,0.5,0.5])/1.2),
                12: (np.array([0.2,0.8,0.4])),
                13: (np.array([0.5,0.2,0.7])),
                14: (np.array([0.2,0.8,0.8])),
                15: (np.array([0.8,0.8,0.8])),
                16: (np.array([0.1,0.9,0.3])),
                17: (np.array([0.4,0,0.6])),
                18: (np.array([0.6,0.1,0.4])),
                19: (np.array([0.4,0.7,0.4])),
                20: (np.array([0.6,0.4,0.6])),
                21: (np.array([0.4,0.3,0.2])),
                22: (np.array([0.6,0.7,0.2])),
                23: (np.array([0.4,0.1,0.8])),
                24: (np.array([0.7,0.3,0.5])),
                25: (np.array([0.2,0.5,0.3])),
                26: (np.array([0.5,0.9,0.1])),
                27: (np.array([0.1,0.4,0.9])),
                28: (np.array([0.8,0.2,0.3])),
                29: (np.array([0.3,0.6,0.7])),
                30: (np.array([0.9,0.5,0.8])),
                31: (np.array([0.5,0.5,0.2])),
                32: (np.array([0.2,0.2,0.5])),
                33: (np.array([0.7,0.7,0.4])),
                34: (np.array([0.4,0.7,0.7])),
                35: (np.array([0.9,0.4,0.2])),
                36: (np.array([0.2,0.4,0.9])),
                37: (np.array([0.6,0.9,0.6])),
                38: (np.array([0.9,0.6,0.9])),
                39: (np.array([0.3,0.3,0.8])),
            }
color_palette = np.array([[0,0,0],
                        [0,182,0],
                        [0,182,182],
                        [182,0,0],
                        [182,0,182],
                        [182,182,0],
                        [170,233,233],
                        [170,0,170],
                        [63,170,0],
                        [170,106,0],
                        [170,0,0],
                        [170,106,106],
                        [51,204,102],
                        [127,51,178],
                        [51,204,204],
                        [204,204,204],
                        [25,229,76],
                        [102,0,153],
                        [153,25,102],
                        [102,178,102],
                        [153,102,153],
                        [102,76,51],
                        [153,178,51],
                        [102,25,204],
                        [178,76,127],
                        [51,127,76],
                        [127,229,25],
                        [25,102,229],
                        [204,51,76],
                        [76,153,178],
                        [229,127,204],
                        [127,127,51],
                        [51,51,127],
                        [178,178,102],
                        [102,178,178],
                        [229,102,51],
                        [51,102,229],
                        [153,229,153],
                        [229,153,229],
                        [76,76,204]], dtype=np.uint8) 
# color_palette = np.array([[  0,   0,   0],  # color,  is_instance_class
#                           [  0, 182,   0],  
#                           [  0, 182, 182],  
#                           [182,   0,   0],  
#                           [182,   0, 182],  # was a problem with 182 -> 82
#                           [182, 182,   0],  
#                           [170, 233, 233],  
#                           [170,   0, 170],  
#                           [170, 170,   0],  
#                           [170, 127,   0],  
#                           [170,   0,   0],  
#                           [170, 127, 127],  
#                           [ 51, 204, 102],  
#                           [128,  51, 178],  # 127 -> 128
#                           [ 51, 204, 204],  
#                           [204, 204, 204],  
#                           [ 26, 229,  77],  # 25 -> 26, 76 -> 77
#                           [102,   0, 153], 
#                           [153,  26, 102],  # 25 -> 26
#                           [102, 178, 102], 
#                           [153, 102, 153], 
#                           [102,  77,  51], # 76 -> 77 
#                           [153, 178,  51], 
#                           [102,  25, 204]], dtype=np.uint8) 

class classes_system:
    """Class system object to store classes and conversion dicts.
    - to_global and from_global dicts should be set in this file below.
    - Import necessary classes_systems from this file and use to_global and from_global for conversion."""
    def __init__(self, classes, inds=None):
        if inds is not None:
            assert len(classes) == len(inds), "classes and inds must have the same length"
        else:
            inds = np.arange(len(classes))
        self.classes = classes
        self.cls_to_ind = {cls: i for i, cls in zip(inds, classes)}
        self.ind_to_cls = {i: cls for i, cls in zip(inds, classes)}
        self.to_global_dict = None
        self.from_global_dict = None

    def __str__(self):
        return f"classes_system(classes={self.classes}, \
                                cls_to_ind={self.cls_to_ind}, \
                                ind_to_cls={self.ind_to_cls}, \
                                to_global_dict={self.to_global_dict}, \
                                from_global_dict={self.from_global_dict})"
    
    def __len__(self):
        return len(self.cls_to_ind)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.cls_to_ind[key]
        elif isinstance(key, int):
            return self.ind_to_cls[key]
        else:
            raise TypeError("key must be either str or int")
        
    def __contains__(self, key) -> bool:
        return key in self.cls_to_ind or key in self.ind_to_cls 
    
    def __str__(self) -> str:
        full_str = "[  -classes: " + str(self.classes)
        full_str += "\n  -cls_to_ind: " + str(self.cls_to_ind)
        full_str += "\n  -ind_to_cls: " + str(self.ind_to_cls)
        full_str += "\n  -to_global_dict: " + str(self.to_global_dict)
        full_str += "\n  -from_global_dict: " + str(self.from_global_dict) 
        full_str += "]\n"
        return full_str
    
    def to_ind(self, cls):
        if isinstance(cls, str):
            return self.cls_to_ind[cls]
        elif isinstance(cls, np.ndarray):
            vectorized_func = np.vectorize(lambda c: self.cls_to_ind[c])
            return vectorized_func(cls)
    
    def to_cls(self, ind):
        if isinstance(ind, int):
            return self.ind_to_cls[ind]
        elif isinstance(ind, np.ndarray):
            vectorized_func = np.vectorize(lambda i: self.ind_to_cls[i])
            return vectorized_func(ind)
        else:
            raise TypeError("Input must be an integer or a numpy array")
        
    def to_global(self, cls) :
        """Must be used with str classes"""
        # logging.info('cls=\n%s',cls)
        # logging.info('self.to_global_dict=\n%s',self.to_global_dict)
        if isinstance(cls, str):
            return self.to_global_dict[cls]
        elif isinstance(cls, np.ndarray):
            vectorized_func = np.vectorize(lambda c: self.to_global_dict[c])
            return vectorized_func(cls)
        
    def from_global(self, cls):
        """Must be used with str classes"""
        if isinstance(cls, str):
            return self.from_global_dict[cls]
        elif isinstance(cls, np.ndarray):
            vectorized_func = np.vectorize(lambda c: self.from_global_dict[c])
            return vectorized_func(cls)
        
    def set_to_global(self, disembiguation_dict={}, background_class="background") -> None:
        """define the to_global dict.
        Provide a dict containing the non direct mappings.
        Example :
        cs = ("drivable",
              "car",
              "other",
              "flower") 
        disembiguation_dict={"drivable":"road"}
        background_class="background"
        -> to_global_dict={"drivable":"road"
                      "car : "car",
                      "other" : "background",
                      'flower': 'background'}
        """
        to_global_dict = {}
        for cls in self.classes:
            if cls in disembiguation_dict:
                assert disembiguation_dict[cls] in global_ext_classes, f"dict[{cls}]={disembiguation_dict[cls]} not in global_ext_classes {global_ext_classes}"
                to_global_dict[cls] = disembiguation_dict[cls]
            elif cls in global_ext_classes:
                to_global_dict[cls] = cls
            elif background_class is not None:
                to_global_dict[cls] = background_class
        self.to_global_dict = to_global_dict
        
    def set_from_global(self, disembiguation_dict={}, background_class="background") -> None:
        """define the from_global dict.
        Provide a dict containing the non direct mappings.
        Example :
        cs = ("drivable",
              "car",
              "other",
              "flower") 
        disembiguation_dict={"road":"drivable",
              "background":"other"}
        -> from_global_dict={"road":"drivable"
                      "car : "car",
                      "background" : "other"}
        """
        from_global_dict = {}
        for cls in global_ext_classes:
            if cls in disembiguation_dict:
                assert disembiguation_dict[cls] in self.cls_to_ind, f"dict[{cls}]={disembiguation_dict[cls]} not in class system {self.classes}"
                from_global_dict[cls] = disembiguation_dict[cls]
            elif cls in self.cls_to_ind:
                from_global_dict[cls] = cls
            elif background_class is not None:
                from_global_dict[cls] = background_class
        self.from_global_dict = from_global_dict
            
global_ext_classes_system = classes_system(global_ext_classes)
global_ext_classes_system.set_to_global(disembiguation_dict={}, background_class="background")
global_ext_classes_system.set_from_global(disembiguation_dict={}, background_class="background")


class classes_systems_converter:
    """Given two classes system, provide methods to convert data from one system to the other."""
    def __init__(self, cs1, cs2):
        self.cs1 = cs1
        self.cs2 = cs2
        self.cs1_to_cs2_dict = {}
        self.cs2_to_cs1_dict = {}
        
        for cls1 in cs1.classes:
            cls2 = cs2.from_global(cs1.to_global(cls1))
            self.cs1_to_cs2_dict[cls1] = cls2
        
        for cls2 in cs2.classes:
            cls1 = cs1.from_global(cs2.to_global(cls2))
            self.cs2_to_cs1_dict[cls2] = cls1
        
        self.cs1_to_cs2_ind_dict = {cs1.to_ind(cls1): cs2.to_ind(cls2) for cls1, cls2 in self.cs1_to_cs2_dict.items()}
        self.cs2_to_cs1_ind_dict = {cs2.to_ind(cls2): cs1.to_ind(cls1) for cls2, cls1 in self.cs2_to_cs1_dict.items()}
    

    def __str__(self):
        return f"cs1_to_cs2_dict={self.cs1_to_cs2_dict},\n \
                cs2_to_cs1_dict={self.cs2_to_cs1_dict},\n \
                cs1_to_cs2_ind_dict={self.cs1_to_cs2_ind_dict},\n \
                cs2_to_cs1_ind_dict={self.cs2_to_cs1_ind_dict})\n"
    
    def cs1_to_cs2(self, labels):
        if isinstance(labels, int):
            return self.cs1_to_cs2_ind_dict[labels]
        elif isinstance(labels, str):
            return self.cs1_to_cs2_dict[labels]
        elif isinstance(labels, np.ndarray):
            if isinstance(labels[0], (int, np.int32, np.int64, np.uint32, np.uint8, np.uint16)):
                return np.array(list(map(self.cs1_to_cs2_ind_dict.get, labels)))
            elif isinstance(labels[0], str):
                return np.array(list(map(self.cs1_to_cs2_dict.get, labels)))
            else:
                raise TypeError(f"Array elements must be either int or str, not {type(labels[0])}")
        else:
            raise TypeError(f"Labels must be either int, str or ndarray, not {type(labels)}")
                
    def cs2_to_cs1(self, labels):
        if isinstance(labels, int):
            return self.cs2_to_cs1_ind_dict[labels]
        elif isinstance(labels, str):
            return self.cs2_to_cs1_dict[labels]
        elif isinstance(labels, np.ndarray):
            if isinstance(labels[0], (int, np.int32, np.int64, np.uint32, np.uint8, np.uint16)):
                return np.array(list(map(self.cs2_to_cs1_ind_dict.get, labels)))
            elif isinstance(labels[0], str):
                return np.array(list(map(self.cs2_to_cs1_dict.get, labels)))
            else:
                raise TypeError(f"Array elements must be either int or str, not {type(labels[0])}")
        else:
            raise TypeError(f"Labels must be either int, str or ndarray, not {type(labels)}")
            
    
##### CLASSES SYSTEM DEFINITIONS #########


#_______________________ ORIGINAL DATASET SYSTEMS _________________________________________________________________
# These are used to read from original annotations

mapillary_nbs_classes = ('background', 
                            'road', 
                            'sidewalk', 
                            'building', 
                            'vegetation', 
                            'terrain', 
                            'sky')
# mapillary_nbs_to_global = {
# }
global_to_mapillary_nbs = {
    'manmade':"building", 
    'parking':"road", 
    'traffic_light':"building", 
    'barrier':"building", 
    'street_light':"building", 
    'fence':"building", 
    'trunk':"vegetation", 
    'wall':"building", 
    'drivable':"road", 
    'other_vehicle':"background", 
    'pedestrian':"background", 
    'car':"background", 
    'vehicle':"background", 
    'truck':"background", 
    'traffic_cone':"building", 
    'bus':"background", 
    'motorcycle':"background", 
    'manhole':"road", 
    'emergency_vehicle':"background", 
    'bicycle':"background", 
    'traffic_sign':"building", 
    'animal':"background", 
    'pole':"building", 
    'ego_vehicle':"background", 
    'road_marking':"road", 
    'construction_vehicle':"background", 
    'pothole':"road", 
    'other_sign':"building"
}
mapillary_nbs_classes_system = classes_system(mapillary_nbs_classes)
mapillary_nbs_classes_system.set_to_global(background_class="background")
mapillary_nbs_classes_system.set_from_global(disembiguation_dict=global_to_mapillary_nbs, background_class="background")



## mapillary_ext
mapillary_ext_classes=('background',
            'road',
            'sidewalk',
            'building',
            'vegetation',
            'terrain',
            'sky',
            'road marking',
            'traffic sign',
            'traffic light',
            'pothole',
            'manhole',
            'street light',
            'pole',
            'vehicle',
            'wall')
mapillary_ext_to_global = {
    "road marking": "road_marking",
    "traffic sign": "traffic_sign",
    "traffic light": "traffic_light",
    "street light": "street_light",
}
global_to_mapillary_ext = {
    "road_marking": "road marking",
     'street_light' : "street light",
     "traffic_sign" : "traffic sign",
     'traffic_light': "traffic light",
     'other_sign': "background", 
     'bicycle': "vehicle", 
     'parking': "road", 
     'animal': "background", 
     'truck': "vehicle", 
     'bus': "vehicle", 
     'fence': "wall", 
     'drivable': "road", 
     'emergency_vehicle': "vehicle", 
     'motorcycle': "vehicle", 
     'other_vehicle': "vehicle",
     'ego_vehicle': "background",
     'pedestrian': "background", 
     'traffic_cone': "building",
     'construction_vehicle': "vehicle",
     'trunk': "vegetation", 
     'car': "vehicle", 
     'barrier': "wall", 
     'manmade': "building", 
     "bicyclist":"background",
     "motorcyclist":"background",
     "stroller":"background",
     "wheelchair":"background",
     "personal_mobility":"background",
}
mapillary_ext_classes_system = classes_system(mapillary_ext_classes)
mapillary_ext_classes_system.set_to_global(disembiguation_dict=mapillary_ext_to_global, background_class="background")
mapillary_ext_classes_system.set_from_global(disembiguation_dict=global_to_mapillary_ext, background_class="background")
waymo_classes = (
    "background",
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_light",
    "traffic_sign",
    "road",
    "sidewalk",
    "lane_marking",
    "curb",
    "pole",
    "traffic_cone",
    "building",
    "vegetation",
    "sky",
    "ground",
    "manmade",
    "water",
    "fence",
    "other",
)
waymo_to_global = {
    "background": "background",
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "bicyclist",
    "traffic_light": "traffic_light",
    "traffic_sign": "traffic_sign",
    "road": "road",
    "sidewalk": "sidewalk",
    "lane_marking": "road_marking",
    "curb": "sidewalk",
    "pole": "pole",
    "traffic_cone": "traffic_cone",
    "building": "building",
    "vegetation": "vegetation",
    "sky": "sky",
    "ground": "terrain",
    "manmade": "manmade",
    "water": "terrain",
    "fence": "fence",
    "other": "background",
}
global_to_waymo = {
    "background": "background", 
    "drivable": "road", 
    "road": "road", 
    "parking": "road", 
    "road_marking": "lane_marking", 
    "other_ground": "ground", 
    "sidewalk": "sidewalk", 
    "terrain": "ground", 
    "sky": "sky", 
    "manmade": "manmade", 
    "building": "building", 
    "traffic_sign": "traffic_sign", 
    "other_sign": "manmade", 
    "traffic_light": "traffic_light", 
    "pothole": "road", 
    "manhole": "road", 
    "street_light": "manmade",
    "pole": "pole",
    "traffic_cone": "traffic_cone",
    "wall": "manmade",
    "barrier": "manmade",
    "fence": "fence",
    "vegetation": "vegetation", 
    "trunk": "vegetation",
    "vehicle": "vehicle", 
    "car": "vehicle",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
    "truck": "vehicle",
    "emergency_vehicle": "vehicle",
    "bus": "vehicle",
    "other_vehicle": "vehicle",
    "construction_vehicle": "vehicle",
    "ego_vehicle": "vehicle",
    "on_rails": "vehicle",
    "pedestrian": "pedestrian",
    "bicyclist": "cyclist",
    "motorcyclist": "vehicle",
    "stroller": "pedestrian",
    "wheelchair": "pedestrian",
    "personal_mobility": "pedestrian",
    "animal": "other", 
}
waymo_classes_system = classes_system(waymo_classes)
waymo_classes_system.set_to_global(disembiguation_dict=waymo_to_global, background_class="background")
waymo_classes_system.set_from_global(disembiguation_dict=global_to_waymo, background_class="background")

nuscenes_global_classes = (
    "noise",
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "flat.driveable_surface",
    "flat.other",
    "flat.sidewalk",
    "flat.terrain",
    # "flat.other",
    "static.manmade",
    "static.other",
    "static.vegetation",
    "vehicle.ego",
    # "static.other",
)
nuscenes_global_to_global = {
    "noise": "background",
    "animal": "animal",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "personal_mobility",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.wheelchair": "wheelchair",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "manmade",
    "movable_object.pushable_pullable": "manmade",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.bicycle_rack": "manmade",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "emergency_vehicle",
    "vehicle.emergency.police": "emergency_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
    "flat.driveable_surface": "road",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "terrain",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "manmade",
    "vehicle.ego": "ego_vehicle",
    "on_rails": "other_vehicle" # not in global, but in global_ext
}
global_to_nuscenes={key: "noise" for key in global_ext_classes}
nuscenes_global_classes_system = classes_system(nuscenes_global_classes)
nuscenes_global_classes_system.set_to_global(disembiguation_dict=nuscenes_global_to_global, background_class="background")
nuscenes_global_classes_system.set_from_global(disembiguation_dict=global_to_nuscenes, background_class="noise")

## nuscenes_uda (ex nuscenes)
# from  https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md and https://www.nuscenes.org/nuscenes#data-annotation
nuscenes_sk19_classes = (
    "noise",
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "flat.driveable_surface",
    "flat.other",
    "flat.sidewalk",
    "flat.terrain",
    # "flat.other",
    "static.manmade",
    "static.other",
    "static.vegetation",
    "vehicle.ego",
    # "static.other",
)
nuscenes_sk19_to_global = {
    "noise": "background",
    "animal": "animal",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "personal_mobility",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.wheelchair": "wheelchair",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "background",
    "movable_object.pushable_pullable": "background",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.bicycle_rack": "manmade",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "emergency_vehicle",
    "vehicle.emergency.police": "emergency_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
    "flat.driveable_surface": "road",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "terrain",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "manmade",
    "vehicle.ego": "ego_vehicle",
    "on_rails": "background" # not in sk19, but in global_ext
}
global_to_nuscenes_sk19 = {"background": "noise",
                        "road": "flat.driveable_surface",
                        "drivable": "flat.driveable_surface",
                        "sidewalk": "flat.sidewalk",
                        "manmade": "static.manmade",
                        "building": "static.manmade",
                        "vegetation": "static.vegetation",
                        "trunk": "static.vegetation",
                        "terrain": "flat.terrain",
                        "parking": "flat.driveable_surface",
                        "road_marking": "flat.driveable_surface",
                        "traffic_sign": "static.manmade",
                        "other_sign": "static.manmade",
                        "traffic_light": "static.manmade",
                        "pothole": "flat.driveable_surface",
                        "manhole": "flat.driveable_surface",
                        "street_light": "static.manmade",
                        "pole": "static.manmade",
                        "car": "vehicle.car",
                        "wall": "static.manmade",
                        "sky": "flat.other",
                        "pedestrian": "human.pedestrian.adult",
                        "bicycle": "vehicle.bicycle",
                        "motorcycle": "vehicle.motorcycle",
                        "truck": "vehicle.truck",
                        "other_vehicle": "vehicle.truck",
                        "vehicle": "vehicle.car",
                        "emergency_vehicle": "vehicle.emergency.ambulance",
                        "bus": "vehicle.bus.rigid",
                        "traffic_cone": "movable_object.trafficcone",
                        "construction_vehicle": "vehicle.construction",
                        "barrier": "movable_object.barrier",
                        "ego_vehicle": "vehicle.ego",
                        "animal": "animal",
                        "bicyclist":"vehicle.bicycle",
                        "motorcyclist":"vehicle.motorcycle",
                        "stroller":"human.pedestrian.stroller",
                        "wheelchair":"human.pedestrian.wheelchair",
                        "personal_mobility":"human.pedestrian.personal_mobility", # e.g. scooter,
                        "other_ground":"noise"
                    }
nuscenes_sk19_classes_system = classes_system(nuscenes_sk19_classes)
nuscenes_sk19_classes_system.set_to_global(disembiguation_dict=nuscenes_sk19_to_global, background_class="background")
nuscenes_sk19_classes_system.set_from_global(disembiguation_dict=global_to_nuscenes_sk19, background_class="noise")


## nuscenes_uda (ex nuscenes)
# from  https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md and https://www.nuscenes.org/nuscenes#data-annotation
nuscenes_uda_classes = (
    "noise",
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "flat.driveable_surface",
    "flat.other",
    "flat.sidewalk",
    "flat.terrain",
    # "flat.other",
    "static.manmade",
    "static.other",
    "static.vegetation",
    "vehicle.ego",
    # "static.other",
)
nuscenes_uda_to_global = {
    "noise": "background",
    "animal": "animal",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "personal_mobility",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.wheelchair": "wheelchair",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "barrier",
    "movable_object.pushable_pullable": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.bicycle_rack": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "emergency_vehicle",
    "vehicle.emergency.police": "emergency_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck",
    "flat.driveable_surface": "road",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "terrain",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "manmade",
    "vehicle.ego": "ego_vehicle",
    "on_rails": "background" # not in uda, but in global_ext
}
global_to_nuscenes_uda = {"background": "noise",
                        "road": "flat.driveable_surface",
                        "drivable": "flat.driveable_surface",
                        "sidewalk": "flat.sidewalk",
                        "manmade": "static.manmade",
                        "building": "static.manmade",
                        "vegetation": "static.vegetation",
                        "trunk": "static.vegetation",
                        "terrain": "flat.terrain",
                        "parking": "flat.driveable_surface",
                        "road_marking": "flat.driveable_surface",
                        "traffic_sign": "static.manmade",
                        "other_sign": "static.manmade",
                        "traffic_light": "static.manmade",
                        "pothole": "flat.driveable_surface",
                        "manhole": "flat.driveable_surface",
                        "street_light": "static.manmade",
                        "pole": "static.manmade",
                        "car": "vehicle.car",
                        "wall": "static.manmade",
                        "sky": "flat.other",
                        "pedestrian": "human.pedestrian.adult",
                        "bicycle": "vehicle.bicycle",
                        "motorcycle": "vehicle.motorcycle",
                        "truck": "vehicle.truck",
                        "other_vehicle": "vehicle.truck",
                        "vehicle": "vehicle.car",
                        "emergency_vehicle": "vehicle.emergency.ambulance",
                        "bus": "vehicle.bus.rigid",
                        "traffic_cone": "movable_object.trafficcone",
                        "construction_vehicle": "vehicle.construction",
                        "barrier": "movable_object.barrier",
                        "ego_vehicle": "vehicle.ego",
                        "animal": "animal",
                        "bicyclist":"vehicle.bicycle",
                        "motorcyclist":"vehicle.motorcycle",
                        "stroller":"human.pedestrian.stroller",
                        "wheelchair":"human.pedestrian.wheelchair",
                        "personal_mobility":"human.pedestrian.personal_mobility", # e.g. scooter,
                        "other_ground":"noise"
                    }
nuscenes_uda_classes_system = classes_system(nuscenes_uda_classes)
nuscenes_uda_classes_system.set_to_global(disembiguation_dict=nuscenes_uda_to_global, background_class="background")
nuscenes_uda_classes_system.set_from_global(disembiguation_dict=global_to_nuscenes_uda, background_class="noise")


nuscenes_dglss_classes = (
    "noise",
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "flat.driveable_surface",
    "flat.other",
    "flat.sidewalk",
    "flat.terrain",
    # "flat.other",
    "static.manmade",
    "static.other",
    "static.vegetation",
    "vehicle.ego",
    # "static.other",
)
nuscenes_dglss_to_global = {
    "noise": "background",
    "animal": "animal",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "personal_mobility",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "stroller",
    "human.pedestrian.wheelchair": "wheelchair",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "barrier",
    "movable_object.pushable_pullable": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "static_object.bicycle_rack": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "emergency_vehicle",
    "vehicle.emergency.police": "emergency_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "other_vehicle",
    "vehicle.truck": "truck",
    "flat.driveable_surface": "road",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "background", # dglss puts kitti's "other_ground" as background, but flat.other is not specified in the table
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "manmade",
    "vehicle.ego": "ego_vehicle",
}
global_to_nuscenes_dglss = {"background": "noise",
                        "road": "flat.driveable_surface",
                        "drivable": "flat.driveable_surface",
                        "sidewalk": "flat.sidewalk",
                        "manmade": "static.manmade",
                        "building": "static.manmade",
                        "vegetation": "static.vegetation",
                        "trunk": "static.vegetation",
                        "terrain": "flat.terrain",
                        "parking": "flat.driveable_surface",
                        "road_marking": "flat.driveable_surface",
                        "traffic_sign": "static.manmade",
                        "other_sign": "static.manmade",
                        "traffic_light": "static.manmade",
                        "pothole": "flat.driveable_surface",
                        "manhole": "flat.driveable_surface",
                        "street_light": "static.manmade",
                        "pole": "static.manmade",
                        "car": "vehicle.car",
                        "wall": "static.manmade",
                        "sky": "flat.other",
                        "pedestrian": "human.pedestrian.adult",
                        "bicycle": "vehicle.bicycle",
                        "motorcycle": "vehicle.motorcycle",
                        "truck": "vehicle.truck",
                        "other_vehicle": "vehicle.truck",
                        "vehicle": "vehicle.car",
                        "emergency_vehicle": "vehicle.emergency.ambulance",
                        "bus": "vehicle.bus.rigid",
                        "traffic_cone": "movable_object.trafficcone",
                        "construction_vehicle": "vehicle.construction",
                        "barrier": "movable_object.barrier",
                        "ego_vehicle": "vehicle.ego",
                        "animal": "animal",
                        "bicyclist":"vehicle.bicycle",
                        "motorcyclist":"vehicle.motorcycle",
                        "stroller":"human.pedestrian.stroller",
                        "wheelchair":"human.pedestrian.wheelchair",
                        "personal_mobility":"human.pedestrian.personal_mobility", # e.g. scooter,
                        "other_ground":"noise",

                        "on_rails": "noise" # not in uda, but in global_ext
                    }
nuscenes_dglss_classes_system = classes_system(nuscenes_dglss_classes)
nuscenes_dglss_classes_system.set_to_global(disembiguation_dict=nuscenes_dglss_to_global, background_class="background")
nuscenes_dglss_classes_system.set_from_global(disembiguation_dict=global_to_nuscenes_dglss, background_class="noise")


semantickitti_global_inds_to_cls = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
semantickitti_global_classes = tuple(semantickitti_global_inds_to_cls.values())
semantickitti_global_inds = tuple(semantickitti_global_inds_to_cls.keys())
semantickitti_global_to_global = {
    "unlabeled": "background",
    "outlier": "background",
    "car": "car",
    "bicycle": "bicycle",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "on-rails": "other_vehicle",
    "truck": "truck",
    "other-vehicle": "other_vehicle",
    "person": "pedestrian",
    "bicyclist": "bicyclist", # following nuscenes convention
    "motorcyclist": "motorcyclist", # following nuscenes convention
    "road": "road",
    "parking": "parking",
    "sidewalk": "sidewalk",
    "other-ground": "other_ground",
    "building": "building", # manmade not included in dglss, but we leave it anyway.
    "fence": "fence", # manmade not included in dglss, but we leave it anyway.
    "other-structure": "manmade",  # manmade not included in dglss, but we leave it anyway.
    "lane-marking": "road_marking",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "pole": "pole", # manmade not included in dglss, but we leave it anyway.
    "traffic-sign": "traffic_sign",
    "other-object": "manmade", # manmade not included in dglss, but we leave it anyway.
    "moving-car": "car",
    "moving-bicyclist": "bicyclist",
    "moving-person": "pedestrian",
    "moving-motorcyclist": "motorcyclist",
    "moving-on-rails": "other_vehicle",
    "moving-bus": "bus",
    "moving-truck": "truck",
    "moving-other-vehicle": "other_vehicle",
}
global_to_semantickitti_global={key: "unlabeled" for key in global_ext_classes}
semantickitti_global_classes_system = classes_system(semantickitti_global_classes, inds=semantickitti_global_inds)
semantickitti_global_classes_system.set_to_global(disembiguation_dict=semantickitti_global_to_global, background_class="background")
semantickitti_global_classes_system.set_from_global(disembiguation_dict=global_to_semantickitti_global, background_class="unlabeled")

## semantickitti_uda (ex semantickitti)
## Original semantickitti classes for conversion into UDA
# from https://github.com/PRBonn/semantic-kitti-api/blob/b269bb43f37e4848af905fe34efd9313bbb15259/config/semantic-kitti.yaml 
semantickitti_uda_inds_to_cls = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
semantickitti_uda_classes = tuple(semantickitti_uda_inds_to_cls.values())
semantickitti_uda_inds = tuple(semantickitti_uda_inds_to_cls.keys())
semantickitti_uda_to_global = {
    "unlabeled": "background",
    "outlier": "background",
    "car": "car",
    "bicycle": "bicycle",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "on-rails": "other_vehicle",
    "truck": "truck",
    "other-vehicle": "other_vehicle",
    "person": "pedestrian",
    "bicyclist": "bicycle", # following nuscenes convention
    "motorcyclist": "motorcycle", # following nuscenes convention
    "road": "road",
    "parking": "parking",
    "sidewalk": "sidewalk",
    "other-ground": "terrain",
    "building": "building", # manmade not included in dglss, but we leave it anyway.
    "fence": "fence", # manmade not included in dglss, but we leave it anyway.
    "other-structure": "manmade",  # manmade not included in dglss, but we leave it anyway.
    "lane-marking": "road",
    "vegetation": "vegetation",
    "trunk": "vegetation",
    "terrain": "terrain",
    "pole": "pole", # manmade not included in dglss, but we leave it anyway.
    "traffic-sign": "background",
    "other-object": "manmade", # manmade not included in dglss, but we leave it anyway.
    "moving-car": "car",
    "moving-bicyclist": "bicycle",
    "moving-person": "pedestrian",
    "moving-motorcyclist": "motorcycle",
    "moving-on-rails": "other_vehicle",
    "moving-bus": "bus",
    "moving-truck": "truck",
    "moving-other-vehicle": "other_vehicle",
}

global_to_semantickitti_uda = {
    "background": "unlabeled",
    "drivable": "road",
    "road": "road",
    "parking": "parking",
    "road_marking": "lane-marking",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "sky": "unlabeled",
    "manmade": "other-structure",
    "building": "building",
    "traffic_sign": "traffic-sign",
    "other_sign": "other-object",
    "traffic_light": "other-structure",
    "pothole": "road",
    "manhole": "road",
    "street_light": "pole",
    "pole": "pole",
    "traffic_cone": "other-object",
    "wall": "other-structure",
    "barrier": "fence",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "vehicle": "car",
    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "emergency_vehicle": "other-vehicle",
    "bus": "bus",
    "other_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "pedestrian": "person",
    "animal": "unlabeled",
    "stroller":"unlabeled",
    "wheelchair":"unlabeled",
    "personal_mobility":"unlabeled",
    "other_ground":"unlabeled",

    "on_rails": "unlabeled" # not in uda, but in global_ext
}
semantickitti_uda_classes_system = classes_system(semantickitti_uda_classes, inds=semantickitti_uda_inds)
semantickitti_uda_classes_system.set_to_global(disembiguation_dict=semantickitti_uda_to_global, background_class="background")
semantickitti_uda_classes_system.set_from_global(disembiguation_dict=global_to_semantickitti_uda, background_class="unlabeled")


# Semantickittit in DLGLSS framework (https://ieeexplore.ieee.org/document/10203512/)
semantickitti_dglss_inds_to_cls = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
semantickitti_dglss_classes = tuple(semantickitti_dglss_inds_to_cls.values())
semantickitti_dglss_inds = tuple(semantickitti_dglss_inds_to_cls.keys())
semantickitti_dglss_to_global = {
    "unlabeled": "background",
    "outlier": "background",
    "car": "car",
    "bicycle": "bicycle",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "on-rails": "other_vehicle",
    "truck": "truck",
    "other-vehicle": "other_vehicle",
    "person": "pedestrian",
    "bicyclist": "bicyclist", # following nuscenes convention
    "motorcyclist": "motorcyclist", # following nuscenes convention
    "road": "road",
    "parking": "road",
    "sidewalk": "sidewalk",
    "other-ground": "background",
    "building": "manmade",
    "fence": "manmade",
    "other-structure": "manmade",
    "lane-marking": "road_marking",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "pole": "manmade",
    "traffic-sign": "manmade",
    "other-object": "manmade",
    "moving-car": "car",
    "moving-bicyclist": "bicyclist",
    "moving-person": "pedestrian",
    "moving-motorcyclist": "motorcyclist",
    "moving-on-rails": "other_vehicle",
    "moving-bus": "bus",
    "moving-truck": "truck",
    "moving-other-vehicle": "other_vehicle",
}

global_to_semantickitti_dglss = { # Useless for UDA exps ?
    "background": "unlabeled",
    "drivable": "road",
    "road": "road",
    "parking": "parking",
    "road_marking": "lane-marking",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "sky": "unlabeled",
    "manmade": "other-structure",
    "building": "building",
    "traffic_sign": "traffic-sign",
    "other_sign": "other-object",
    "traffic_light": "other-structure",
    "pothole": "road",
    "manhole": "road",
    "street_light": "pole",
    "pole": "pole",
    "traffic_cone": "other-object",
    "wall": "other-structure",
    "barrier": "fence",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "vehicle": "car",
    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "emergency_vehicle": "other-vehicle",
    "bus": "bus",
    "other_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "pedestrian": "person",
    "animal": "unlabeled",
    "bicyclist":"unlabeled",
    "motorcyclist":"unlabeled",
    "stroller":"unlabeled",
    "wheelchair":"unlabeled",
    "personal_mobility":"unlabeled", # e.g. scooter
    "on_rails": "unlabeled" # not in uda, but in global_ext
}

semantickitti_dglss_classes_system = classes_system(semantickitti_dglss_classes, inds=semantickitti_dglss_inds)
semantickitti_dglss_classes_system.set_to_global(disembiguation_dict=semantickitti_dglss_to_global, background_class="background")
semantickitti_dglss_classes_system.set_from_global(disembiguation_dict=global_to_semantickitti_dglss, background_class="unlabeled")


# Semantickitti classes for sk19 conversion
# Use this as the base classes system before conversion to sk19
# e.g. dataset_name=semantickitti_sk19 and then in dense GT: new_classes_system=sk19
# semantickitti_sk19 -> sk19 has been checked by hand from https://github.com/PRBonn/semantic-kitti-api/blob/b269bb43f37e4848af905fe34efd9313bbb15259/config/semantic-kitti.yaml
semantickitti_sk19_inds_to_cls = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
semantickitti_sk19_classes = tuple(semantickitti_sk19_inds_to_cls.values())
semantickitti_sk19_inds = tuple(semantickitti_sk19_inds_to_cls.keys())
semantickitti_sk19_to_global = {
    "unlabeled": "background",
    "outlier": "background",
    "car": "car",
    "bicycle": "bicycle",
    "bus": "other_vehicle",
    "motorcycle": "motorcycle",
    "on-rails": "on_rails",
    "truck": "truck",
    "other-vehicle": "other_vehicle",
    "person": "pedestrian",
    "bicyclist": "bicyclist", # following nuscenes convention
    "motorcyclist": "motorcyclist", # following nuscenes convention
    "road": "road",
    "parking": "parking",
    "sidewalk": "sidewalk",
    "other-ground": "other_ground",
    "building": "building",
    "fence": "fence",
    "other-structure": "background",
    "lane-marking": "road_marking",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "pole": "pole",
    "traffic-sign": "traffic_sign",
    "other-object": "background",
    "moving-car": "car",
    "moving-bicyclist": "bicyclist",
    "moving-person": "pedestrian",
    "moving-motorcyclist": "motorcyclist",
    "moving-on-rails": "other_vehicle",
    "moving-bus": "other_vehicle",
    "moving-truck": "truck",
    "moving-other-vehicle": "other_vehicle",
}
global_to_semantickitti_sk19 = { # Useless for UDA exps ?
                                
    "background": "unlabeled",
    "drivable": "road",
    "road": "road",
    "parking": "parking",
    "road_marking": "lane-marking",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "other_ground": "other-ground",
    "sky": "unlabeled",
    "manmade": "building",
    "building": "building",
    "traffic_sign": "traffic-sign",
    "other_sign": "other-object",
    "traffic_light": "other-structure",
    "pothole": "road",
    "manhole": "road",
    "street_light": "pole",
    "pole": "pole",
    "traffic_cone": "other-object",
    "wall": "other-structure",
    "barrier": "fence",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "vehicle": "car",
    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "emergency_vehicle": "other-vehicle",
    "bus": "bus",
    "other_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "pedestrian": "person",
    "animal": "unlabeled",
    "bicyclist":"bicyclist",
    "motorcyclist":"motorcyclist",
    "stroller":"unlabeled",
    "wheelchair":"unlabeled",
    "personal_mobility":"unlabeled", # e.g. scooter
    "on_rails": "unlabeled" # not in uda, but in global_ext
}

semantickitti_sk19_classes_system = classes_system(semantickitti_sk19_classes, inds=semantickitti_sk19_inds)
semantickitti_sk19_classes_system.set_to_global(disembiguation_dict=semantickitti_sk19_to_global, background_class="background")
semantickitti_sk19_classes_system.set_from_global(disembiguation_dict=global_to_semantickitti_sk19, background_class="unlabeled")

# Semantickitti classes for SK->NS test evaluation
semantickitti_ns16_inds_to_cls = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
semantickitti_ns16_classes = tuple(semantickitti_ns16_inds_to_cls.values())
semantickitti_ns16_inds = tuple(semantickitti_ns16_inds_to_cls.keys())
semantickitti_ns16_to_global = {
    "unlabeled": "background",
    "outlier": "background",
    "car": "car",
    "bicycle": "bicycle",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "on-rails": "on_rails",
    "truck": "truck",
    "other-vehicle": "other_vehicle",
    "person": "pedestrian",
    "bicyclist": "bicyclist", # following nuscenes convention
    "motorcyclist": "motorcyclist", # following nuscenes convention
    "road": "road",
    "parking": "parking",
    "sidewalk": "sidewalk",
    "other-ground": "other_ground",
    "building": "building",
    "fence": "fence",
    "other-structure": "background",
    "lane-marking": "road_marking",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "pole": "pole",
    "traffic-sign": "traffic_sign",
    "other-object": "background",
    "moving-car": "car",
    "moving-bicyclist": "bicyclist",
    "moving-person": "pedestrian",
    "moving-motorcyclist": "motorcyclist",
    "moving-on-rails": "other_vehicle",
    "moving-bus": "bus",
    "moving-truck": "truck",
    "moving-other-vehicle": "other_vehicle",
}
global_to_semantickitti_ns16 = { # Useless for UDA exps ?
                                
    "background": "unlabeled",
    "drivable": "road",
    "road": "road",
    "parking": "parking",
    "road_marking": "lane-marking",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "other_ground": "other-ground",
    "sky": "unlabeled",
    "manmade": "building",
    "building": "building",
    "traffic_sign": "traffic-sign",
    "other_sign": "other-object",
    "traffic_light": "other-structure",
    "pothole": "road",
    "manhole": "road",
    "street_light": "pole",
    "pole": "pole",
    "traffic_cone": "other-object",
    "wall": "other-structure",
    "barrier": "fence",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "vehicle": "car",
    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "emergency_vehicle": "other-vehicle",
    "bus": "bus",
    "other_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "pedestrian": "person",
    "animal": "unlabeled",
    "bicyclist":"bicyclist",
    "motorcyclist":"motorcyclist",
    "stroller":"unlabeled",
    "wheelchair":"unlabeled",
    "personal_mobility":"unlabeled", # e.g. scooter
    "on_rails": "unlabeled" # not in uda, but in global_ext
}

semantickitti_ns16_classes_system = classes_system(semantickitti_ns16_classes, inds=semantickitti_ns16_inds)
semantickitti_ns16_classes_system.set_to_global(disembiguation_dict=semantickitti_ns16_to_global, background_class="background")
semantickitti_ns16_classes_system.set_from_global(disembiguation_dict=global_to_semantickitti_ns16, background_class="unlabeled")


# semantickitti_rare_classes for conversion to rare classes
semantickitti_rare_classes_inds_to_cls = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}
semantickitti_rare_classes = tuple(semantickitti_rare_classes_inds_to_cls.values())
semantickitti_rare_classes_inds = tuple(semantickitti_rare_classes_inds_to_cls.keys())
semantickitti_rare_classes_to_global = {
    "unlabeled": "background",
    "outlier": "background",
    "car": "background",
    "bicycle": "background",
    "bus": "background",
    "motorcycle": "background",
    "on-rails": "background",
    "truck": "background",
    "other-vehicle": "background",
    "person": "background",
    "bicyclist": "background", 
    "motorcyclist": "background", 
    "road": "background",
    "parking": "background",
    "sidewalk": "background",
    "other-ground": "background",
    "building": "background",
    "fence": "background",
    "other-structure": "background",
    "lane-marking": "road_marking", # class 1
    "vegetation": "background",
    "trunk": "background",
    "terrain": "background",
    "pole": "background",
    "traffic-sign": "traffic_sign", # class 2
    "other-object": "background",
    "moving-car": "background",
    "moving-bicyclist": "background",
    "moving-person": "background",
    "moving-motorcyclist": "background",
    "moving-on-rails": "background",
    "moving-bus": "background",
    "moving-truck": "background",
    "moving-other-vehicle": "background",
}
global_to_semantickitti_rare_classes = { # Arbitrarily filled, doesn't matter a priori  
    "background": "unlabeled",
    "drivable": "road",
    "road": "road",
    "parking": "parking",
    "road_marking": "lane-marking",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "other_ground": "other-ground",
    "sky": "unlabeled",
    "manmade": "building",
    "building": "building",
    "traffic_sign": "traffic-sign",
    "other_sign": "other-object",
    "traffic_light": "other-structure",
    "pothole": "road",
    "manhole": "road",
    "street_light": "pole",
    "pole": "pole",
    "traffic_cone": "other-object",
    "wall": "other-structure",
    "barrier": "fence",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "vehicle": "car",
    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "emergency_vehicle": "other-vehicle",
    "bus": "bus",
    "other_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "pedestrian": "person",
    "animal": "unlabeled",
    "bicyclist":"bicyclist",
    "motorcyclist":"motorcyclist",
    "stroller":"unlabeled",
    "wheelchair":"unlabeled",
    "personal_mobility":"unlabeled", # e.g. scooter
    "on_rails": "unlabeled" # not in uda, but in global_ext
}

semantickitti_rare_classes_system = classes_system(semantickitti_rare_classes, inds=semantickitti_rare_classes_inds)
semantickitti_rare_classes_system.set_to_global(disembiguation_dict=semantickitti_rare_classes_to_global, background_class="background")
semantickitti_rare_classes_system.set_from_global(disembiguation_dict=global_to_semantickitti_rare_classes, background_class="unlabeled")


## synlidar_sk19
# from https://drive.google.com/drive/folders/18Fbb1JHV9tjFTTCb0V5IPg1eLo3thw5v
# synlidar_sk19 -> sk19 has been checked by hand, exactly following SynLidar_sk19's mapping (from annotations.yaml in link above) 

synlidar_sk19_classes = (
    "unlabeled",
    "car",
    "pick-up",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "other-vehicle",
    "road",
    "sidewalk",
    "parking",
    "other-ground",
    "female",
    "male",
    "kid",
    "crowd",  # multiple person that are very close
    "bicyclist",
    "motorcyclist",
    "building",
    "other-structure",
    "vegetation",
    "trunk",
    "terrain",
    "traffic-sign",
    "pole",
    "traffic-cone",
    "fence",
    "garbage-can",
    "electric-box",
    "table",
    "chair",
    "bench",
    "other-object"
)
synlidar_sk19_to_global = {
    "unlabeled": "background",
    "car": "car",
    "pick-up": "truck",  
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "other-vehicle": "other_vehicle",
    "road": "road",
    "sidewalk": "sidewalk",
    "parking": "parking",
    "other-ground": "other_ground",
    "female": "pedestrian",
    "male": "pedestrian",
    "kid": "pedestrian",
    "crowd": "pedestrian",  # Multiple people, mapped to pedestrian
    "bicyclist": "bicyclist",
    "motorcyclist": "motorcyclist",
    "building": "building",
    "other-structure": "background",  # Maps to unlabeled in sk19
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "traffic-sign": "traffic_sign",
    "pole": "pole",
    "traffic-cone": "background",  # Maps to unlabeled in sk19
    "fence": "fence",
    "garbage-can": "background",  # Maps to unlabeled in sk19
    "electric-box": "background",  # Maps to unlabeled in sk19
    "table": "background",  # Maps to unlabeled in sk19
    "chair": "background",  # Maps to unlabeled in sk19
    "bench": "background",  # Maps to unlabeled in sk19
    "other-object": "background"  # Maps to unlabeled in sk19
}

global_to_synlidar_sk19 = {
    "background": "unlabeled",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "other_vehicle": "other-vehicle",
    "road": "road",
    "sidewalk": "sidewalk",
    "parking": "parking",
    "other_ground": "other-ground",
    "pedestrian": "male",  # Default to male, could be any of the person classes
    "bicyclist": "bicyclist",
    "motorcyclist": "motorcyclist",
    "building": "building",
    "manmade": "other-structure",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "traffic_sign": "traffic-sign",
    "other_sign": "traffic-sign",
    "pole": "pole", 
    "fence": "fence",
    "drivable": "road",
    "road_marking": "road",
    "traffic_light": "pole",  # No direct match
    "pothole": "road",  # No direct match
    "manhole": "road",  # No direct match
    "street_light": "pole",  # No direct match
    "traffic_cone": "traffic-cone",
    "wall": "other-structure",
    "barrier": "fence",  # Closest match
    "vehicle": "car",  # Default to car
    "emergency_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "stroller": "unlabeled",
    "wheelchair": "unlabeled",
    "personal_mobility": "unlabeled",
    "animal": "unlabeled",
    "sky": "unlabeled",
    "on_rails": "unlabeled" # not in uda, but in global_ext
}
synlidar_sk19_classes_system = classes_system(synlidar_sk19_classes)
synlidar_sk19_classes_system.set_to_global(disembiguation_dict=synlidar_sk19_to_global, background_class="background")
synlidar_sk19_classes_system.set_from_global(disembiguation_dict=global_to_synlidar_sk19, background_class="unlabeled")

## synlidar_dglss
# SynLidar classes remapped for conversion to the dglss target system
synlidar_dglss_classes = (
    "unlabeled",
    "car",
    "pick-up",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "other-vehicle",
    "road",
    "sidewalk",
    "parking",
    "other-ground",
    "female",
    "male",
    "kid",
    "crowd",
    "bicyclist",
    "motorcyclist",
    "building",
    "other-structure",
    "vegetation",
    "trunk",
    "terrain",
    "traffic-sign",
    "pole",
    "traffic-cone",
    "fence",
    "garbage-can",
    "electric-box",
    "table",
    "chair",
    "bench",
    "other-object"
)
# Follow semantickitti_dglss mapping philosophy to global
synlidar_dglss_to_global = {
    "unlabeled": "background",
    "car": "car",
    "pick-up": "truck",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "other-vehicle": "other_vehicle",
    "road": "road",
    "sidewalk": "sidewalk",
    "parking": "road",  # dglss uses drivable-surface, treated as road in global
    "other-ground": "background",  # per dglss, mapped to background
    "female": "pedestrian",
    "male": "pedestrian",
    "kid": "pedestrian",
    "crowd": "pedestrian",
    "bicyclist": "bicyclist",
    "motorcyclist": "motorcyclist",
    "building": "manmade",
    "other-structure": "manmade",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "terrain": "terrain",
    "traffic-sign": "manmade",
    "pole": "manmade",
    "traffic-cone": "background",
    "fence": "manmade",
    "garbage-can": "manmade",
    "electric-box": "manmade",
    "table": "manmade",
    "chair": "manmade",
    "bench": "manmade",
    "other-object": "manmade",
}

global_to_synlidar_dglss = {
    "background": "unlabeled",
    "drivable": "road",
    "road": "road",
    "parking": "parking",
    "road_marking": "road",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "other_ground": "other-ground",
    "sky": "unlabeled",
    "manmade": "other-structure",
    "building": "building",
    "traffic_sign": "traffic-sign",
    "other_sign": "traffic-sign",
    "traffic_light": "other-structure",
    "pothole": "road",
    "manhole": "road",
    "street_light": "pole",
    "pole": "pole",
    "traffic_cone": "traffic-cone",
    "wall": "other-structure",
    "barrier": "fence",
    "fence": "fence",
    "vegetation": "vegetation",
    "trunk": "trunk",
    "vehicle": "car",
    "car": "car",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "emergency_vehicle": "other-vehicle",
    "bus": "bus",
    "other_vehicle": "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "unlabeled",
    "pedestrian": "male",
    "animal": "unlabeled",
    "bicyclist": "unlabeled",
    "motorcyclist": "unlabeled",
    "stroller": "unlabeled",
    "wheelchair": "unlabeled",
    "personal_mobility": "unlabeled",
    "on_rails": "unlabeled"
}

synlidar_dglss_classes_system = classes_system(synlidar_dglss_classes)
synlidar_dglss_classes_system.set_to_global(disembiguation_dict=synlidar_dglss_to_global, background_class="background")
synlidar_dglss_classes_system.set_from_global(disembiguation_dict=global_to_synlidar_dglss, background_class="unlabeled")


#_______________________ TARGET CLASSES SYSTEMS _________________________________________________________________
# These are used to convert into for inference

## uda
uda_classes=(
    'background', 
    'car', 
    'bicycle',
    'motorcycle',
    'truck',
    'other vehicle',
    'person',
    'drivable',
    'sidewalk', 
    'terrain', 
    'vegetation', 
    'manmade',) # neither in T-UDA nor LIDAR-UDA
    # "fence")
uda_to_global = {
    "other vehicle": "other_vehicle",
    "person": "pedestrian",
}
global_to_uda = {
    'vehicle' : "car", 
    'parking' : "drivable", 
    'emergency_vehicle' : "other vehicle", 
    'traffic_light' : "manmade", 
    'road' : "drivable", 
    'wall' : "manmade", 
    'traffic_sign' : "manmade", 
    'fence' : "manmade", 
    'pole' : "manmade", 
    'other_vehicle' : "other vehicle", 
    'barrier' : "manmade", 
    'ego_vehicle' : "background", 
    'pedestrian' : "person", 
    'building' : "manmade", 
    'street_light' : "manmade", 
    'animal' : "background", 
    'sky' : "background", 
    'traffic_cone' : "manmade", 
    'road_marking' : "drivable", 
    'manhole' : "drivable", 
    'other_sign' : "manmade", 
    'trunk' : "vegetation", 
    'bus' : "other vehicle", 
    'pothole' : "drivable", 
    'construction_vehicle' : "other vehicle",
    "bicyclist":"background",
    "motorcyclist":"background",
    "stroller":"background",
    "wheelchair":"background",
    "personal_mobility":"background",
    "other_ground":"background",

    "on_rails": "background" # not in uda, but in global_ext
}
uda_classes_system = classes_system(uda_classes)
uda_classes_system.set_to_global(disembiguation_dict=uda_to_global, background_class="background")
uda_classes_system.set_from_global(disembiguation_dict=global_to_uda, background_class="background")
# logging.info('uda_classes_system.cls_to_ind=\n%s',uda_classes_system.cls_to_ind)
# logging.info('uda_classes_system.to_global_dict=\n%s',uda_classes_system.to_global_dict)
# logging.info('uda_classes_system.from_global_dict=\n%s',uda_classes_system.from_global_dict)



## dglss
dglss_classes=(
    'background', 
    'car', 
    'bicycle',
    'motorcycle',
    'truck',
    'other-vehicle',
    'pedestrian',
    'drivable-surface',
    'sidewalk', 
    'walkable', 
    'vegetation') 
    # 'manmade',) # neither in T-dglss nor LIDAR-dglss
    # "fence")
dglss_to_global = {
    "other-vehicle": "other_vehicle",
    "drivable-surface": "drivable",
    "walkable": "terrain",
}
global_to_dglss = {
    "drivable": "drivable-surface", # drivable is more general than road. Use road when possible.
    "road": "drivable-surface", 
    "parking": "drivable-surface",
    "road_marking": "drivable-surface",
    "terrain": "walkable",
    "sky": "background",
    "manmade": "background", 
    "building": "background",
    "traffic_sign": "background",
    "other_sign": "background",
    "traffic_light": "background",
    "pothole": "background",
    "manhole": "background",
    "street_light": "background",
    "pole": "background",
    "traffic_cone": "background",
    "wall": "background",
    "barrier": "background",
    "fence": "background",
    "vegetation": "vegetation",
    "trunk": "vegetation",
    # "vehicle", 
    "emergency_vehicle": "background",
    "bus": "other-vehicle",
    "other_vehicle" : "other-vehicle",
    "construction_vehicle": "other-vehicle",
    "ego_vehicle": "background",
    "animal": "background",
    "bicyclist":"background",
    "motorcyclist":"background",
    "stroller":"background",
    "wheelchair":"background",
    "personal_mobility":"background",
    "animal":"background",
    "other_ground":"background",

    "on_rails": "background" # not in uda, but in global_ext
}
dglss_classes_system = classes_system(dglss_classes)
dglss_classes_system.set_to_global(disembiguation_dict=dglss_to_global, background_class="background")
dglss_classes_system.set_from_global(disembiguation_dict=global_to_dglss, background_class="background")
     

## sk19
# SemanticKITTI 19 classes, used in multiple papers e.g. SALUDA 
# To use sk19 for semantickitti dataset, use semantickitti_sk19_classes_system
# from https://github.com/PRBonn/semantic-kitti-api/blob/b269bb43f37e4848af905fe34efd9313bbb15259/config/semantic-kitti.yaml 
# From "learning_map" (only 19 classes + unlabeled), used in multiple papers e.g. SALUDA
# CHECKED semantickitti->sk19 using the link above
sk19_classes = (
    "background", # or "outlier" or "other-structure" or "other-object"
    "car", # "moving-car"
    "bicycle",
    "motorcycle",
    "truck", # "moving-truck"
    "other_vehicle", # or "bus" or "bus", # "moving-on-rails" # "moving-other-vehicle" # "moving-bus"
    "pedestrian", # "moving-person"
    "bicyclist", # "moving-bicyclist"
    "motorcyclist", # "moving-motorcyclist"
    "road", # "lane-marking"
    "parking",
    "sidewalk",
    "other_ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic_sign", # or "other-object"
)
sk19_to_global = {
}
global_to_sk19 = {
    "background": "background", # general garbage class for all classes that are not in the system and outliers
    "drivable": "road", # drivable is more general than road. Use road when possible.
    "road": "road", 
    "parking": "parking",
    "road_marking": "road",
    "other_ground": "other_ground",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "sky": "background",
    "manmade": "building", # manmade is more general than all manmade classes, use the specific class when possible.
    "building": "building",
    "traffic_sign": "traffic_sign",
    "other_sign": "background",
    "traffic_light": "background",
    "pothole":"road",
    "manhole":"road",
    "street_light":"pole",
    "pole":"pole",
    "traffic_cone":"building",
    "wall":"building",
    "barrier":"fence",
    "fence":"fence",
    "vegetation":"vegetation", # vegetation is more general than trunk. Use trunk when possible.
    "trunk":"trunk",
    "vehicle":"car", # vehicle is more general than all vehicle classes. Use the specific class when possible.
    "car":"car",
    "bicycle":"bicycle",
    "motorcycle":"motorcycle",
    "truck":"truck",
    "emergency_vehicle":"other_vehicle",
    "bus":"other_vehicle",
    "other_vehicle":"other_vehicle",
    "construction_vehicle":"other_vehicle",
    "ego_vehicle":"background",
    "on_rails":"other_vehicle",
    "pedestrian":"pedestrian",
    "bicyclist":"bicyclist",
    "motorcyclist":"motorcyclist",
    "stroller":"background",
    "wheelchair":"background",
    "personal_mobility":"background",
    "animal":"background",
}
sk19_classes_system = classes_system(sk19_classes)
sk19_classes_system.set_to_global(disembiguation_dict=sk19_to_global, background_class="background")
sk19_classes_system.set_from_global(disembiguation_dict=global_to_sk19, background_class="background")

## ns16
# nuScenes 16 classes used in official test benchmark
# from https://www.nuscenes.org/lidar-segmentation?externalData=all&mapData=all&modalities=Any
ns16_classes = (
        "background",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation"
)
ns16_to_global = {
}
global_to_ns16 = {
    "background": "background",
    "drivable": "driveable_surface", 
    "road": "driveable_surface", 
    "parking": "driveable_surface",
    "road_marking": "driveable_surface",
    "other_ground": "other_flat",
    "sidewalk": "sidewalk",
    "terrain": "terrain",
    "sky": "background",
    "manmade": "manmade",
    "building": "manmade",
    "traffic_sign": "manmade",
    "other_sign": "background",
    "traffic_light": "background",
    "pothole":"driveable_surface",
    "manhole":"driveable_surface",
    "street_light":"manmade",
    "pole":"manmade",
    "traffic_cone":"traffic_cone",
    "wall":"manmade",
    "barrier":"barrier",
    "fence":"barrier",
    "vegetation":"vegetation",
    "trunk":"vegetation",
    "vehicle":"car", 
    "car":"car",
    "bicycle":"bicycle",
    "motorcycle":"motorcycle",
    "truck":"truck",
    "emergency_vehicle":"background", # from https://www.nuscenes.org/lidar-segmentation?externalData=all&mapData=all&modalities=Any
    "bus":"bus",
    "other_vehicle":"background",
    "construction_vehicle":"background",
    "ego_vehicle":"background",
    "on_rails":"background",
    "pedestrian":"pedestrian",
    "bicyclist":"pedestrian",
    "motorcyclist":"pedestrian",
    "stroller":"background",
    "wheelchair":"background",
    "personal_mobility":"background",
    "animal":"background",
}
ns16_classes_system = classes_system(ns16_classes)
ns16_classes_system.set_to_global(disembiguation_dict=ns16_to_global, background_class="background")
ns16_classes_system.set_from_global(disembiguation_dict=global_to_ns16, background_class="background")

## rare
# Rare classes system, used for "rare classes" experiment (seg_3D_by_PC2D as pretraining for IV24)
# To use rare for semantickitti dataset, use semantickitti_rare_classes_system
rare_classes = (
    "background", # or "outlier" or "other-structure" or "other-object"
    "traffic_sign", # or "other-object"
    "road_marking",
)
rare_to_global = {
    "background":"background",
    "traffic_sign":"traffic_sign",
    "road_marking":"road_marking",
}
global_to_rare = {
    "background": "background",
    "drivable": "background",
    "road": "background",
    "parking": "background",
    "road_marking": "road_marking",
    "other_ground": "background",
    "sidewalk": "background",
    "terrain": "background",
    "sky": "background",
    "manmade": "background",
    "building": "background",
    "traffic_sign": "traffic_sign",
    "other_sign": "background",
    "traffic_light": "background",
    "pothole": "background",
    "manhole": "background",
    "street_light": "background",
    "pole": "background",
    "traffic_cone": "background",
    "wall": "background",
    "barrier": "background",
    "fence": "background",
    "vegetation": "background",
    "trunk": "background",
    "vehicle": "background",
    "car": "background",
    "bicycle": "background",
    "motorcycle": "background",
    "truck": "background",
    "emergency_vehicle": "background",
    "bus": "background",
    "other_vehicle": "background",
    "construction_vehicle": "background",
    "ego_vehicle": "background",
    "on_rails": "background",
    "pedestrian": "background",
    "bicyclist": "background",
    "motorcyclist": "background",
    "stroller": "background",
    "wheelchair": "background",
    "personal_mobility": "background",
    "animal": "background",
}
rare_classes_system = classes_system(rare_classes)
rare_classes_system.set_to_global(disembiguation_dict=rare_to_global, background_class="background")
rare_classes_system.set_from_global(disembiguation_dict=global_to_rare, background_class="background")



    
if __name__ == "__main__":
    
        parser = argparse.ArgumentParser(description='Print different classes between 2 systems')
        parser.add_argument('--cs1', help='', required=True)
        parser.add_argument('--cs2', help='', required=True)
        args = parser.parse_args()
    
        assert args.cs1 in systems, f"args.cs1={args.cs1} not in systems {systems}"
        assert args.cs2 in systems, f"args.cs2={args.cs2} not in systems {systems}"
        
        cs1 = globals()[f"{args.cs1}_classes_system"]
        cs2 = globals()[f"{args.cs2}_classes_system"]
        
        classes1 = set(cs1.classes)
        classes2 = set(cs2.classes)
        logging.info("\nClasses in both systems:")
        for cls in sorted(classes1 & classes2):
            logging.info("  %s", cls)
            
        logging.info("\nClasses in %s but not in %s:", args.cs1, args.cs2)
        for cls in sorted(classes1 - classes2):
            logging.info("  %s", cls)
            
        logging.info("\nClasses in %s but not in %s:", args.cs2, args.cs1)
        for cls in sorted(classes2 - classes1):
            logging.info("  %s", cls)

        cs_converter = classes_systems_converter(cs1, cs2)
        
        logging.info("\nConverting from %s to %s:", args.cs1, args.cs2)
        logging.info("cs_converter.cs1_to_cs2_dict:")
        for k,v in sorted(cs_converter.cs1_to_cs2_dict.items()):
            logging.info("  %s --> %s", k, v)
            
        logging.info("\nConverting from %s to %s:", args.cs2, args.cs1)
        logging.info("cs_converter.cs2_to_cs1_dict:")
        for k,v in sorted(cs_converter.cs2_to_cs1_dict.items()):
            logging.info("  %s --> %s", k, v)
            
        logging.info("\nConverting indices from %s to %s:", args.cs1, args.cs2)
        logging.info("cs_converter.cs1_to_cs2_ind_dict:")
        for k,v in sorted(cs_converter.cs1_to_cs2_ind_dict.items()):
            logging.info("  %s --> %s", k, v)
            
        logging.info("\nConverting indices from %s to %s:", args.cs2, args.cs1)
        logging.info("cs_converter.cs2_to_cs1_ind_dict:")
        for k,v in sorted(cs_converter.cs2_to_cs1_ind_dict.items()):
            logging.info("  %s --> %s", k, v)


    
        # cs1 = globals()["nuscenes_uda_classes_system"]
        # cs2 = globals()["semantickitti_sk19_classes_system"]
        # cs3 = globals()["waymo_classes_system"]
        # cs_global = globals()["global_ext_classes_system"]
        
        # cs1_to_global = classes_systems_converter(cs1, cs_global)
        # unique_global_classes = sorted(set(cs1_to_global.cs1_to_cs2_dict.values()))
        # logging.info("nuscenes classes:" + ',' + ",".join(unique_global_classes))
            
        # cs2_to_global = classes_systems_converter(cs2, cs_global)
        # unique_global_classes = sorted(set(cs2_to_global.cs1_to_cs2_dict.values()))
        # logging.info("semantickitti classes:" + ',' + ",".join(unique_global_classes))
            
        # cs3_to_global = classes_systems_converter(cs3, cs_global)
        # unique_global_classes = sorted(set(cs3_to_global.cs1_to_cs2_dict.values()))
        # logging.info("waymo classes: " + ',' + ",".join(unique_global_classes))

        # # # LaTeX table generation
        # # unique_cs1 = set(cs1_to_global.cs1_to_cs2_dict.values())
        # unique_cs1 = set(nuscenes_uda_to_global.values())
        # unique_cs2 = set(semantickitti_sk19_to_global.values())
        # unique_cs3 = set(waymo_to_global.values())
        
        # all_classes = sorted(unique_cs1 | unique_cs2 | unique_cs3)
        
        # print(r"\begin{table}[]")
        # print(r"\begin{tabular}{|l|c|c|c|}")
        # print(r"\hline")
        # print(r"Class & NuScenes & SemanticKITTI & Waymo \\ \hline")
        
        # for cls in all_classes:
        #     row = [cls.replace("_", r"\_")]
        #     row.append(r"\checkmark" if cls in unique_cs1 else "")
        #     row.append(r"\checkmark" if cls in unique_cs2 else "")
        #     row.append(r"\checkmark" if cls in unique_cs3 else "")
        #     print(" & ".join(row) + r" \\ \hline")
            
        # print(r"\end{tabular}")
        # print(r"\end{table}")
