# roadDeformityDetection
pythonanywhere flask API to detect potholes and segment cracks on road images

Input : POST input image

Output : returns JPEG image

Pothole Detection
 - RCNN Model @ http://sriyar.pythonanywhere.com/detect/rcnn
 - SSD Model @ http://sriyar.pythonanywhere.com/detect/ssd
Curl Command : 
curl -k -X POST -F 'image=@/path/to/image' -v http://sriyaR.pythonanywhere.com/detect/rcnn > tested_rcnn.jpg

Crack Segmentation
 - UNet Model @ http://sriyar.pythonanywhere.com/segment
Curl Command : 
curl -k -X POST -F 'image=@/path/to/image' -v http://sriyaR.pythonanywhere.com/segment > tested_segment.jpg
