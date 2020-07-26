import cv2
import numpy as np

# Load Yolo
net=cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Classes
classes=[]
with open("coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()]

# Define Output Layers    
layer_names=net.getLayerNames()
outputlayers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# Load Image
img=cv2.imread("d2.jpg")
img=cv2.resize(img,None,fx=0.4,fy=0.3)
height,width,channel=img.shape

# Plot the image
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#detecting objects using blob
blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),True,crop=False)

# Plot all the blob images
# for b in blob:
#     for n,img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Send blob as input
net.setInput(blob)

#Object Detection
outs=net.forward(outputlayers)

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

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
font=cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        label=str(classes[class_ids[i]])
        color=COLORS[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,6)
        cv2.putText(img,label,(x,y+30),font,1,(255,0,0),2)

cv2.imshow("Detected_Images",img)
cv2.waitKey(0)
cv2.destroyAllWindows()