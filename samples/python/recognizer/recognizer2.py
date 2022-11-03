'''
    * Copyright (C) 2011-2020 Doubango Telecom <https://www.doubango.org>
    * File author: Mamadou DIOP (Doubango Telecom, France).
    * License: For non commercial use only.
    * Source code: https://github.com/DoubangoTelecom/ultimateALPR-SDK
    * WebSite: https://www.doubango.org/webapps/alpr/


    https://github.com/DoubangoTelecom/ultimateALPR/blob/master/SDK_dist/samples/c++/recognizer/README.md
    Usage: 
        recognizer.py \
            --image <path-to-image-with-plate-to-recognize> \
            [--assets <path-to-assets-folder>] \
            [--charset <recognition-charset:latin/korean/chinese>] \
            [--tokenfile <path-to-license-token-file>] \
            [--tokendata <base64-license-token-data>]
    Example:
        recognizer.py \
            --image C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dist/assets/images/lic_us_1280x720.jpg \
            --charset "latin" \
            --assets C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dist/assets \
            --tokenfile C:/Projects/GitHub/ultimate/ultimateALPR/SDK_dev/tokens/windows-iMac.lic
'''

import ultimateAlprSdk
import argparse
import json
import os.path
import numpy
import cv2
import math
import time
import dlib

TAG = "[PythonRecognizer] "

# Defines the default JSON configuration. More information at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html
JSON_CONFIG = {
    "debug_level": "info",
    "debug_write_input_image_enabled": False,
    "debug_internal_data_path": ".",
    
    "num_threads": -1,
    "gpgpu_enabled": True,
    "max_latency": -1,

    "klass_vcr_gamma": 1.5,
    
    "detect_roi": [0, 0, 0, 0],
    "detect_minscore": 0.1,

    "car_noplate_detect_min_score": 0.8,
    
    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.28,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,
    
    "recogn_rectify_enabled": True,
    "recogn_minscore": 0.3,
    "recogn_score_type": "min"
}

IMAGE_TYPES_MAPPING = { 
        'RGB': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24,
        'RGBA': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGBA32,
        'L': ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_Y
}

# Load image
def load_pil_image(path):
    from PIL import Image, ExifTags, ImageOps
    import traceback
    pil_image = Image.open(path)
    img_exif = pil_image.getexif()
    ret = {}
    orientation  = 1
    try:
        if img_exif:
            for tag, value in img_exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                ret[decoded] = value
            orientation  = ret["Orientation"]
    except Exception as e:
        print(TAG + "An exception occurred: {}".format(e))
        traceback.print_exc()

    if orientation > 1:
        pil_image = ImageOps.exif_transpose(pil_image)

    if pil_image.mode in IMAGE_TYPES_MAPPING:
        imageType = IMAGE_TYPES_MAPPING[pil_image.mode]
    else:
        raise ValueError(TAG + "Invalid mode: %s" % pil_image.mode)

    return pil_image, imageType

# Load image from opencv video frame
def load_pil_image_from_opencv_frame(opencv_image):
    from PIL import Image, ExifTags, ImageOps
    import traceback

    # convert from openCV2 to PIL.
    color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted)

    # Get image exif data
    img_exif = pil_image.getexif()
    ret = {}
    orientation  = 1
    try:
        if img_exif:
            for tag, value in img_exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                ret[decoded] = value
            orientation  = ret["Orientation"]
    except Exception as e:
        print(TAG + "An exception occurred: {}".format(e))
        traceback.print_exc()

    if orientation > 1:
        pil_image = ImageOps.exif_transpose(pil_image)

    if pil_image.mode in IMAGE_TYPES_MAPPING:
        imageType = IMAGE_TYPES_MAPPING[pil_image.mode]
    else:
        raise ValueError(TAG + "Invalid mode: %s" % pil_image.mode)

    return pil_image, imageType

# Check result
def checkResult(operation, result):
    if not result.isOK():
        print(TAG + operation + ": failed -> " + result.phrase())
        assert False
    else:
        print(TAG + operation + ": OK -> " + result.json())

        print("")
        print("")

        #print(json.dumps(alpr_output, indent=4))

        #plate_output = alpr_output.get("duration")

        #number = plate_output[1]

        #dict_output = json.loads(result.json())

        #with open('json_out.json', 'w', encoding ='utf8') as json_file:
        #    json.dump(dict_output, json_file, ensure_ascii = False)
        
        #with open('json_out.json') as f:
        #    alpr_output = json.load(f)
        #extracted_output = alpr_output['plates'][0]['warpedBox']
        #print(alpr_output.get("plates")[0])


        #print("The results is: ", plate_output)

# Convert bounding box format from (xmin, ymin, xmax, ymax) to (xmin, ymin, w, h)
def convert_bb_to_wh(xmin, ymin, xmax, ymax):

    w = xmax - xmin
    h = ymax - ymin

    return tuple(map(int, (xmin, ymin, w, h)))


# Get the parameters from alpr result
def detect_car_and_plate(alpr_output_result):

    # Get the json output from ALPR as dictionary
    alpr_output = json.loads(alpr_output_result.json())

    detection_results = []

    # Number of vehicles detected
    num_cars_detected = alpr_output_result.numCars()

    # If no car found, i.e. detection_result is empty list
    if num_cars_detected < 1:
        return []

    # Loop through all the plates found
    for plate_idx in range(num_cars_detected):
        # Slice into the car_box
        car_box = list(map(int, alpr_output['plates'][plate_idx]['car']['warpedBox']))
        car_x_min = car_box[0]
        car_y_min = car_box[1]
        car_x_max = car_box[4]
        car_y_max = car_box[5]
        car_box_converted = convert_bb_to_wh(car_x_min, car_y_min, car_x_max, car_y_max)

        # Slice into car_confidence in percentage as float format
        car_confidence = round(alpr_output['plates'][plate_idx]['car']['confidence'], 1)

        # Slice into plate_box, plate_confidence, plate_number, ocr_confidence. None if no plates found.
        plate_box = alpr_output['plates'][plate_idx].get('warpedBox', None)

        # If there is a number plate
        if plate_box:
            plate_box = list(map(int, plate_box))
            plate_x_min = plate_box[0]
            plate_y_min = plate_box[1]
            plate_x_max = plate_box[4]
            plate_y_max = plate_box[5]
            plate_box_converted = (plate_x_min, plate_y_min, plate_x_max, plate_y_max)
            plate_confidence = round(alpr_output['plates'][plate_idx]['confidences'][1], 1)
            plate_number = alpr_output['plates'][plate_idx]['text']
            ocr_confidence = round(alpr_output['plates'][plate_idx]['confidences'][0], 1)
        
        # If no number plate was found
        else:
            plate_box_converted = None
            plate_confidence = None
            plate_number = None
            ocr_confidence = None
        
        
        vehicle = { 'car_box': car_box_converted,
                    'car_confidence': car_confidence,
                    'plate_box': plate_box_converted,
                    'plate_confidence': plate_confidence,
                    'plate_number': plate_number,
                    'ocr_confidence': ocr_confidence }

        detection_results.append(vehicle)





                # # Slice into the car_box, plate_number, and plate_box
                # car_box = list(map(int, alpr_output['plates'][plate_idx]['car']['warpedBox']))
                # plate_number = alpr_output['plates'][plate_idx]['text']
                # plate_box = list(map(int, alpr_output['plates'][plate_idx]['warpedBox']))

                # # Slice into car_confidence, and plate_confidence in percentage as float format
                # car_confidence = round(alpr_output['plates'][plate_idx]['car']['confidence'], 1)
                # plate_confidence = round(alpr_output['plates'][plate_idx]['confidences'][1], 1)
                # ocr_confidence = round(alpr_output['plates'][plate_idx]['confidences'][0], 1)

    return detection_results

# Function to estimate speed using ppm
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 85.0
    d_meters = d_pixels / ppm
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 30
    speed = d_meters * fps * 3.6
    return speed



# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This is the recognizer sample using python language
    """)

    parser.add_argument("--image", required=True, help="Path to the image with ALPR data to recognize")
    parser.add_argument("--assets", required=False, default="../../../assets", help="Path to the assets folder")
    parser.add_argument("--charset", required=False, default="latin", help="Defines the recognition charset (a.k.a alphabet) value (latin, korean, chinese...)")
    parser.add_argument("--car_noplate_detect_enabled", required=False, default=False, help="Whether to detect and return cars with no plate")
    parser.add_argument("--ienv_enabled", required=False, default=False, help="Whether to enable Image Enhancement for Night-Vision (IENV). More info about IENV at https://www.doubango.org/SDKs/anpr/docs/Features.html#image-enhancement-for-night-vision-ienv. Default: true for x86-64 and false for ARM.")
    parser.add_argument("--openvino_enabled", required=False, default=True, help="Whether to enable OpenVINO. Tensorflow will be used when OpenVINO is disabled")
    parser.add_argument("--openvino_device", required=False, default="CPU", help="Defines the OpenVINO device to use (CPU, GPU, FPGA...). More info at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html#openvino-device")
    parser.add_argument("--npu_enabled", required=False, default=True, help="Whether to enable NPU (Neural Processing Unit) acceleration")
    parser.add_argument("--klass_lpci_enabled", required=False, default=False, help="Whether to enable License Plate Country Identification (LPCI). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#license-plate-country-identification-lpci")
    parser.add_argument("--klass_vcr_enabled", required=False, default=False, help="Whether to enable Vehicle Color Recognition (VCR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-color-recognition-vcr")
    parser.add_argument("--klass_vmmr_enabled", required=False, default=False, help="Whether to enable Vehicle Make Model Recognition (VMMR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-make-model-recognition-vmmr")
    parser.add_argument("--klass_vbsr_enabled", required=False, default=False, help="Whether to enable Vehicle Body Style Recognition (VBSR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-body-style-recognition-vbsr")
    parser.add_argument("--tokenfile", required=False, default="", help="Path to license token file")
    parser.add_argument("--tokendata", required=False, default="", help="Base64 license token data")

    args = parser.parse_args()

    

    # Check if image exist
    if not os.path.isfile(args.image):
        raise OSError(TAG + "File doesn't exist: %s" % args.image)



    # Update JSON options using values from the command args
    JSON_CONFIG["assets_folder"] = args.assets
    JSON_CONFIG["charset"] = args.charset
    JSON_CONFIG["car_noplate_detect_enabled"] = (args.car_noplate_detect_enabled == "True")
    JSON_CONFIG["ienv_enabled"] = (args.ienv_enabled == "True")
    JSON_CONFIG["openvino_enabled"] = (args.openvino_enabled == "True")
    JSON_CONFIG["openvino_device"] = args.openvino_device
    JSON_CONFIG["npu_enabled"] = (args.npu_enabled == "True")
    JSON_CONFIG["klass_lpci_enabled"] = (args.klass_lpci_enabled == "True")
    JSON_CONFIG["klass_vcr_enabled"] = (args.klass_vcr_enabled == "True")
    JSON_CONFIG["klass_vmmr_enabled"] = (args.klass_vmmr_enabled == "True")
    JSON_CONFIG["klass_vbsr_enabled"] = (args.klass_vbsr_enabled == "True")
    JSON_CONFIG["license_token_file"] = args.tokenfile
    JSON_CONFIG["license_token_data"] = args.tokendata



    # Initialize the engine
    checkResult("Init", 
                ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG))
               )

    video = cv2.VideoCapture(args.image)




    # variables
    vehicleBoxColour = (0, 255, 0)
    plateBoxColour = (200, 0, 255)
    frameCounter = 0
    currentCarID = 0
    currentPlateID = 0
    # for calculating frame rate
    fps = 0
    
    carTracker = {}
    plateTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    carConfidence = {}
    plateLocation = {}
    plateConfidence = {}
    plateNumber = {}
    ocrConfidence = {}
    speed = [None] * 1000
    
    # Write output to video file
    #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))
    out = cv2.VideoWriter('traffic_oncoming_1_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))


    while True:
        start_time = time.time()
        rc, opencv_image = video.read()

        if type(opencv_image) == type(None):
            break

        # Decode the image and extract type
        image, imageType = load_pil_image_from_opencv_frame(opencv_image)
        width, height = image.size

        
        
        opencv_image = cv2.resize(opencv_image, (width, height))
        resultImage = opencv_image.copy()
        
        frameCounter = frameCounter + 1
        
        carIDtoDelete = []

        # Clear out vehicle boxes with poor tracking
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(opencv_image)
            
            if trackingQuality < 7:
                carIDtoDelete.append(carID)
                
        for carID in carIDtoDelete:
            print ('Removing carID ' + str(carID) + ' from list of vehicle trackers.')
            print ('Removing carID ' + str(carID) + ' previous location.')
            print ('Removing carID ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        plateIDtoDelete = []
        
        # Clear out plate boxes with poor tracking
        for plateID in plateTracker.keys():
            plateTrackingQuality = plateTracker[plateID].update(opencv_image)
            
            if plateTrackingQuality < 7:
                plateIDtoDelete.append(plateID)
                
        for plateID in plateIDtoDelete:
            print ('Removing plateID ' + str(plateID) + ' from list of plate trackers.')
            print ('Removing plateID ' + str(plateID) + ' location.')
            plateTracker.pop(plateID, None)
            plateLocation.pop(plateID, None)
            plateConfidence.pop(plateID, None)
            plateNumber.pop(plateID, None)
            ocrConfidence.pop(plateID, None)
        
        if not (frameCounter % 15):
            # Get the result from ALPR
            alpr_output_result = ultimateAlprSdk.UltAlprSdkEngine_process(
                    imageType,
                    image.tobytes(), # type(x) == bytes
                    width,
                    height,
                    0, # stride
                    1 # exifOrientation (already rotated in load_image -> use default value: 1)
                )
            

            # Get the output from alpr_output in specified variables
            detection_results = detect_car_and_plate(alpr_output_result)
            
            for detection_result in detection_results:
                # Get the car bounding box
                car_x, car_y, car_w, car_h = detection_result['car_box']

                # Average center position of vehicle
                car_x_bar = car_x + 0.5 * car_w
                car_y_bar = car_y + 0.5 * car_h

                matchCarID = None

                # Loop through all vehicles being tracked
                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                
                    if ((t_x <= car_x_bar <= (t_x + t_w)) and (t_y <= car_y_bar <= (t_y + t_h)) and (car_x <= t_x_bar <= (car_x + car_w)) and (car_y <= t_y_bar <= (car_y + car_h))):
                        matchCarID = carID

                # If a new vehicle is found
                if matchCarID is None:
                    print ('Creating new vehicle tracker ' + str(currentCarID))
                    
                    # Track vehicle
                    car_tracker = dlib.correlation_tracker()
                    car_tracker.start_track(opencv_image, dlib.rectangle(car_x, car_y, car_x + car_w, car_y + car_h))

                    # Update dictionaries with new detection result  
                    carTracker[currentCarID] = car_tracker
                    carLocation1[currentCarID] = [car_x, car_y, car_w, car_h]
                    carConfidence[currentCarID] = detection_result['car_confidence']

                    currentCarID += 1

                # If the vehicle detected is already being tracked
                else:
                    # Update dictionaries with new detection result
                    carConfidence[matchCarID] = detection_result['car_confidence']

                # If a plate is detected
                if detection_result['plate_box']:
                    # Get the plate bounding box
                    plate_x_min, plate_y_min, plate_x_max, plate_y_max = detection_result['plate_box']
                    
                    # Average center position of plate
                    plate_x_bar = plate_x_min + 0.5 * (plate_x_max - plate_x_min)
                    plate_y_bar = plate_y_min + 0.5 * (plate_y_max - plate_y_min)

                    matchPlateID = None

                    # Loop through all plates being tracked
                    for plateID in plateTracker.keys():
                        plateTrackedPosition = plateTracker[plateID].get_position()
                        
                        t_x = int(plateTrackedPosition.left())
                        t_y = int(plateTrackedPosition.top())
                        t_w = int(plateTrackedPosition.width())
                        t_h = int(plateTrackedPosition.height())
                        
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h
                    
                        if ((t_x <= plate_x_bar <= (t_x + t_w)) and (t_y <= plate_y_bar <= (t_y + t_h)) and (plate_x_min <= t_x_bar <= (plate_x_max)) and (plate_y_min <= t_y_bar <= (plate_y_max))):
                            matchPlateID = plateID
                    


                    # If a new plate is found
                    if matchPlateID is None:
                        print ('Creating new plate tracker ' + str(currentPlateID))

                        # Track plate
                        plate_tracker = dlib.correlation_tracker()
                        plate_tracker.start_track(opencv_image, dlib.rectangle(detection_result['plate_box'][0], detection_result['plate_box'][1], detection_result['plate_box'][2], detection_result['plate_box'][3]))

                        # Update dictionaries with new detection result
                        plateTracker[currentPlateID] = plate_tracker
                        plateLocation[currentPlateID] = detection_result['plate_box']
                        plateConfidence[currentPlateID] = detection_result['plate_confidence']
                        plateNumber[currentPlateID] = detection_result['plate_number']
                        ocrConfidence[currentPlateID] = detection_result['ocr_confidence']
                        

                        currentPlateID += 1

                    # If the plate detected is already being tracked
                    else:
                        # Update dictionaries with new detection result
                        plateLocation[matchPlateID] = detection_result['plate_box']
                        plateConfidence[matchPlateID] = detection_result['plate_confidence']
                        
                        
                        # If confidence for plate number has increased
                        if detection_result['ocr_confidence'] > ocrConfidence[matchPlateID]:
                            ocrConfidence[matchPlateID] = detection_result['ocr_confidence']
                            plateNumber[matchPlateID] = detection_result['plate_number']
        
        #cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

        # Loop through vehicles being tracked
        for carID in carTracker.keys():
            # Tracked position of vehicle
            trackedPosition = carTracker[carID].get_position()
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            
            # Draw bounding box for vehicle detected
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), vehicleBoxColour, 2)

            # Write the confidence for the vehicle
            cv2.putText(resultImage, " Vehicle (" + str(carConfidence[carID]) + "%)", (t_x, t_y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, vehicleBoxColour, 1)

            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        # Loop through plates being tracked
        for plateID in plateTracker.keys():
            # Tracked position of plate
            plateTrackedPosition = plateTracker[plateID].get_position()
            plate_t_x = int(plateTrackedPosition.left())
            plate_t_y = int(plateTrackedPosition.top())
            plate_t_w = int(plateTrackedPosition.width())
            plate_t_h = int(plateTrackedPosition.height())

            # Draw bounding box for plate
            cv2.rectangle(resultImage, (plate_t_x, plate_t_y), (plate_t_x + plate_t_w, plate_t_y + plate_t_h), plateBoxColour, 2)

            # Write the confidence for the plate
            cv2.putText(resultImage, " Plate (" + str(plateConfidence[plateID]) + "%)", (plate_t_x, plate_t_y - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, plateBoxColour, 1)

            # Write the plate number
            cv2.putText(resultImage, str(plateNumber[plateID]) + " (" + str(ocrConfidence[plateID]) + "%)", (plate_t_x, plate_t_y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, plateBoxColour, 1)
        
        end_time = time.time()
        
        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)
        
        #cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Compute speed
        for i in carLocation1.keys():	
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
        
                # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                carLocation1[i] = [x2, y2, w2, h2]
        
                # print 'new previous location: ' + str(carLocation1[i])
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 100 and y1 <= 900:     # originally: y1 >= 275 and y1 <= 285
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    #if y1 > 275 and y1 < 285:
                    if speed[i] != None and y1 >= 100:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1), int(y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, vehicleBoxColour, 1)
                        
                    
                    #print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                    #else:
                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        #print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
        cv2.imshow('result', resultImage)

        # Write the frame into the file 'output.avi'
        out.write(resultImage)


        if cv2.waitKey(33) == 27:
            break
    
    cv2.destroyAllWindows()



    # Recognize/Process
    # Please note that the first time you call this function all deep learning models will be loaded 
    # and initialized which means it will be slow. In your application you've to initialize the engine
    # once and do all the recognitions you need then, deinitialize it.
    # checkResult("Process",
    #             ultimateAlprSdk.UltAlprSdkEngine_process(
    #                 imageType,
    #                 image.tobytes(), # type(x) == bytes
    #                 width,
    #                 height,
    #                 0, # stride
    #                 1 # exifOrientation (already rotated in load_image -> use default value: 1)
    #             )
    #     )
    
    # alpr_result = ultimateAlprSdk.UltAlprSdkEngine_process(
    #                 imageType,
    #                 image.tobytes(), # type(x) == bytes
    #                 width,
    #                 height,
    #                 0, # stride
    #                 1 # exifOrientation (already rotated in load_image -> use default value: 1)
    #             )

    # # Process the output from ALPR
    # alpr_output = json.loads(ultimateAlprSdk.UltAlprSdkEngine_process(
    #                 imageType,
    #                 image.tobytes(), # type(x) == bytes
    #                 width,
    #                 height,
    #                 0, # stride
    #                 1 # exifOrientation (already rotated in load_image -> use default value: 1)
    #             ).json())

    # # Number of car plates detected
    # num_of_plates = len(alpr_output['plates'])

    # # Loop through all the plates found
    # for plate_idx in range(num_of_plates):
    #     # Slice into the car_box, plate_number, and plate_box
    #     car_box = list(map(int, alpr_output['plates'][plate_idx]['car']['warpedBox']))
    #     plate_number = alpr_output['plates'][plate_idx]['text']
    #     plate_box = list(map(int, alpr_output['plates'][plate_idx]['warpedBox']))

    #     # Slice into car_confidence, and plate_confidence in percentage as float format
    #     car_confidence = round(alpr_output['plates'][plate_idx]['car']['confidence'], 1)
    #     plate_confidence = round(alpr_output['plates'][plate_idx]['confidences'][1], 1)
    #     ocr_confidence = round(alpr_output['plates'][plate_idx]['confidences'][0], 1)

    #     # Print the results obtained from alpr
    #     print(f"The num_of_plates is: {num_of_plates}")
    #     print(f"The car_box is: {car_box}")
    #     print(f"The plate_number is: {plate_number}")
    #     print(f"The plate_box is: {plate_box}")
    #     print(f"The car_confidence is: {car_confidence}")
    #     print(f"The plate_confidence is: {plate_confidence}")
    #     print(f"The ocr_confidence is: {ocr_confidence}")

        

    #     # Add bounding box for vehicle
    #     cv_img = cv2.rectangle(cv_img, (car_box[0], car_box[1]), (car_box[4], car_box[5]), (0, 0, 255), 2)

    #     # Add text for vehicle
    #     cv_img = cv2.putText(cv_img, "Vehicle" + " (" + str(car_confidence) + "%)", (car_box[0], car_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    #     # Add bounding box for number plate
    #     cv_img = cv2.rectangle(cv_img, (plate_box[0], plate_box[1]), (plate_box[4], plate_box[5]), (255, 0, 255), 2)

    #     # Add text for plate
    #     cv_img = cv2.putText(cv_img, "Plate"  + " (" + str(plate_confidence) + "%)", (plate_box[0], plate_box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 255), 1)

    #     # Add text for plate's alphanumerics
    #     cv_img = cv2.putText(cv_img, plate_number  + " (" + str(ocr_confidence) + "%)", (plate_box[0], plate_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 255), 1)

    # # Display image
    # cv2.imshow("alpr_out", cv_img)

    # Press any key to exit
    input("\nPress Enter to exit...\n") 

    # DeInit the engine
    checkResult("DeInit", 
                ultimateAlprSdk.UltAlprSdkEngine_deInit()
               )
    
    