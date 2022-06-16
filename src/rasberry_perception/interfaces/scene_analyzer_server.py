import ros_numpy

from rasberry_perception.interfaces.registry import DETECTION_REGISTRY
from rasberry_perception.interfaces.default import BaseDetectionServer
from rasberry_perception.msg import Detections, ServiceStatus, Detection, RegionOfInterest, SegmentOfInterest
from rasberry_perception.srv import GetDetectorResultsRequest, GetDetectorResultsResponse

import time 
import rospy
import os
from sensor_msgs.msg import Image

#general imports
import os, sys, numpy as np, pickle, logging
import pickle            
# detectron imports
from detectron2.config           import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2                  import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data             import MetadataCatalog, DatasetCatalog
import cv2

@DETECTION_REGISTRY.register_detection_backend("scene_analyzer_server")  # (1)
class SceneAnalyzerServer(BaseDetectionServer):  # (2)
    # These args are passed from ros parameters when running the backend
    def __init__(self, model_path="/home/arshad/robot_highway/data_ws/src/raspberry_fruit_mapping/berry_tracker/models/fp_model.pth", metadata_path="/home/arshad/robot_highway/data_ws/src/raspberry_fruit_mapping/berry_tracker/models/metadata.pkl", cfg_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" ): 


        # initialize mask predictor
        self.busy = True
        self.scene_analyser_predictor = None
        if model_path == "" : 
            raise ("Please provide scene analyser model file absolute path") 
        elif metadata_path == "" : 
            raise ("Please proide scene analyser meta file absolute path")
        elif cfg_path == "": 
            raise ("Please proide scene analyser config name")
        elif not os.path.exists( model_path ): 
            raise ("'model_path' does not exists" )
        elif not os.path.exists( metadata_path ) : 
            raise ("'metadata_path' does not exists" )
        else: 
            # Do initialisation code here
            # self.scene/_analyser_predictor = DetectronCocoObjectDetection( )
            self.scene_analyser_predictor = MasksPredictor( model_path, cfg_path, metadata_path )

        self.busy = False 
        rospy.loginfo("Initializing the scene analyzer server...")
        time.sleep(1)
        BaseDetectionServer.__init__(self)  # Spins the server and waits for requests!

    def get_detector_results(self, request):  # (3)
        if self.busy:  # Example of other status responses
            return GetDetectorResultsResponse(status=ServiceStatus(BUSY=True))
        
        # if self.scene_analyser_predictor == None: 
        #     return GetDetectorResultsResponse(status=ServiceStatus(ERROR=True))
        
        # Populate a detections message
        detections = Detections()

        rospy.loginfo("Processing the request from client here..")
        detections = self.scene_analyser_predictor.get_predictions( request.image )
        # i.e. detections = image_to_results_function(image=ros_numpy.numpify(request.image))
        return GetDetectorResultsResponse(status=ServiceStatus(OKAY=True), results=detections)

    

"""
Repurposed from scene_analyser from FastPick
"""

class MasksPredictor:

    def __init__(self, model_file, config_file, metadata_file, num_classes  = 3 , accuracy = 0.01):
        
        self.counter = 1
        self.accuracy = accuracy 
        self.scale=1.0

        self.metadata=self.get_metadata(metadata_file)
        self.cfg = self.init_config(model_file, config_file, num_classes)

        print ( self.metadata)
        print ( self.cfg.DATASETS.TRAIN[0] )
        try:
            self.predictor=DefaultPredictor(self.cfg)
        except Exception as e:
            logging.error(e)
            print(e)
            raise Exception(e)

    def init_config(self, model_file, config_file, num_classes):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except Exception as e:
            logging.error(e)
            print(e)
            raise Exception(e)

        cfg.MODEL.WEIGHTS = model_file #os.path.join(model_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 1.strawberry, 2. Canopy, 3. Rigid Structure
        return cfg

    def get_metadata(self,metadata_file):

        #metadata file has name of classes, it is created to avoid having custom dataset and taking definitions
        # from annotations, instead. It has structure of MetaDataCatlog output of detectron2
        try:
            file = open(metadata_file, 'rb')
        except Exception as e:
            logging.error(e)
            print(e)
            raise Exception(e)

        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self, ros_rgb_image ):
        

        np_rgb_image = ros_numpy.numpify( ros_rgb_image )
        outputs = self.predictor( np_rgb_image )

        v = Visualizer(np_rgb_image[:, :, ::-1], self.metadata , scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        pred_classes = outputs["instances"].get_fields()["pred_classes"].to("cpu").numpy()
        scores = outputs["instances"].get_fields()["scores"].to("cpu").numpy()
        pred_masks = outputs["instances"].get_fields()["pred_masks"].to("cpu").numpy()
        pred_boxes = outputs["instances"].get_fields()["pred_boxes"].to("cpu").tensor.numpy()

        prediction_output = {
            "pred_classes" : pred_classes,
            "scores" : scores,
            "pred_masks" : pred_masks,
            "pred_boxes" : pred_boxes,
        }            
 

        output_image = out.get_image()[:, :, ::-1].astype(np.uint8)
        output_filepath = "/home/arshad/Desktop/outputs/output_{}.png".format( self.counter )
        cv2.imwrite( output_filepath , np.concatenate( (cv2.cvtColor(np_rgb_image, cv2.COLOR_RGB2BGR), cv2.cvtColor(output_image.copy(), cv2.COLOR_RGB2BGR) ), axis=0 ) ) 
        self.counter += 1

        detections = Detections()
        detections.camera_frame = ros_numpy.msgify( Image, output_image, encoding=ros_rgb_image.encoding)

        for detected_object, score in enumerate(scores): 
            
            if (score > self.accuracy and pred_classes[detected_object] == 0 ):

                detection = Detection()
                
                roi = RegionOfInterest()
                soi = SegmentOfInterest()
                roi.x1, roi.y1, roi.x2, roi.y2 = pred_boxes[detected_object]

                mask = np.where( pred_masks[detected_object]  )
                soi.x , soi.y = list( mask[1] ) , list( mask[0] ) 
            
                detection.roi = roi
                detection.seg_roi = soi
                detection.class_name = "Strawberry"
                detection.confidence = score
                detections.objects.append( detection )

        return detections