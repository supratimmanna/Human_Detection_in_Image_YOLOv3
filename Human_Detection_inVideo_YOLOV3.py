import cv2
import numpy as np
import time
# Load Yolo
#net=cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
net=cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Classes
classes=[]
with open("coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()]

# Define Output Layers    
layer_names=net.getLayerNames()
outputlayers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
font=cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture('test.mp4')

out_video = cv2.VideoWriter(
    'output_human.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
starting_time=time.time()
frame_id=0

while (True):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.resize(frame, (640, 480))
        height,width,channel=frame.shape
        #out.write(frame.astype('uint8'))
        
        #detecting objects using blob. (320,320) can be used instead of (608,608)
        blob=cv2.dnn.blobFromImage(frame,1/255,(320,320),(0,0,0),True,crop=False)
        
        net.setInput(blob)
    
        #Object Detection
        outs=net.forward(outputlayers)
    
        # Evaluate class ids, confidence score and bounding boxes
        class_ids=[]
        confidences=[]
        boxes=[]
    
        for out in outs:
            for detection in out:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]
                if confidence>0.5:
                    # Object Detected
                    center_x=int(detection[0]*width)
                    center_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)
                    #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    
                    # Rectangle Co-ordinates
                    x=int(center_x-w/2)
                    y=int(center_y-h/2)
                    
                    boxes.append([x,y,w,h]) #Put all rectangle boxes
                    confidences.append(float(confidence)) #How confidence that the oject is detected
                    class_ids.append(class_id) #Name of the detected object
                    
                    # Non-max Suppression
                    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    
        # Draw final bounding boxes
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h=boxes[i]
                label=str(classes[class_ids[i]])
                color=COLORS[i]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+" "+str(round(confidences[i],3)),(x,y+30),font,1,(255,0,0),2)
                
        cv2.imshow("Detected_Images",frame)
        #frame = Image.fromarray(frame)
        out_video.write(frame.astype('uint8'))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
