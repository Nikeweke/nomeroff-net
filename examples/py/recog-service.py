import os
import numpy as np
import sys
import matplotlib.image as mpimg
import json


# change this property
NOMEROFF_NET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet import  filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
textDetector = TextDetector.get_static_module("eu")()
textDetector.load("latest")




def recognizePlate(imagePath):
  # Detect numberplate
  print("START RECOGNIZING")
  img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../images/' + imagePath)

  img = mpimg.imread(img_path)
  NP = nnet.detect([img])

  # Generate image mask.
  cv_img_masks = filters.cv_img_mask(NP)

  # Detect points.
  arrPoints = rectDetector.detect(cv_img_masks)
  zones = rectDetector.get_cv_zonesBGR(img, arrPoints)

  # find standart
  regionIds, stateIds, countLines = optionsDetector.predict(zones)
  regionNames = optionsDetector.getRegionLabels(regionIds)
  
  # find text with postprocessing by standart  
  textArr = textDetector.predict(zones)
  textArr = textPostprocessing(textArr, regionNames)

  # my_json_string = json.dumps(textArr)
  print(json.dumps({'result': textArr}))
  return json.dumps({'result': textArr})


# ================================================> HTTP-SERVER
import urllib.parse
from urllib.parse import urlparse 
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

  def do_GET(self):
    try:
      query = urlparse(self.path).query
      query_components = dict(qc.split("=") for qc in query.split("&"))
      imagePath = query_components["image"]

      result = recognizePlate(imagePath)

      self.send_response(200)
      self.end_headers()
      self.wfile.write(bytes(result, 'utf8'))

    except:
      self.send_response(400)
      self.end_headers()
      self.wfile.write(bytes(json.dumps({'error': 'Error occured'}), 'utf8'))
   



httpd = HTTPServer(('0.0.0.0', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
