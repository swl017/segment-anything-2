import torch
import numpy as np
import PIL
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import cv_bridge

import matplotlib.pyplot as plt
import copy

import rospy
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray

class ImagePredictor:
    def __init__(self, sam2_checkpoint, model_cfg):
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.sam2_model = None
        self.predictor = None
        self.bridge = cv_bridge.CvBridge()
        self.image = None
        self.detection = None
        self.image_subscriber = rospy.Subscriber("/fpv_image", Image, self.image_callback)
        self.detection_subscriber = rospy.Subscriber("/yolov7/yolov7", Detection2DArray, self.detection_callback)
        self.visualization_publisher = rospy.Publisher("/sam2_image", Image, queue_size=1)
        self.visualization_publisher_compressed = rospy.Publisher("/sam2_image/compressed", CompressedImage, queue_size=1)

    def load_model(self):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def show_mask(self, mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)

    def draw_mask(self, image, mask, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = np.zeros((h, w, 4), dtype=np.uint8)
        mask_image[:, :, :3] = image
        mask_image[:, :, 3] = mask * 255
        
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(mask_image, contours, -1, (*color[:3], 255), thickness=2)
        
        return mask_image
    
    def draw_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        result_image = image.copy()
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_image = self.draw_mask(result_image, mask, borders=borders)
            
            # Blend the mask with the result image
            alpha = 0.5
            result_image = cv2.addWeighted(result_image, 1, mask_image[:, :, :3], alpha, 0)
            
            if box_coords is not None:
                # Draw bounding box
                cv2.rectangle(result_image, (int(box_coords[0]), int(box_coords[1])), 
                            (int(box_coords[2]), int(box_coords[3])), (0, 255, 0), 2)
            
            if len(scores) > 0:
                x_text = int(box_coords[0])
                y_text = max(15, int(box_coords[1] - 10))
                text = f"Mask {i+1}, Score: {score:.3f}"
                cv2.putText(result_image, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return mask_image
        # return result_image
    
    def publish_image(self, image):
        vis_msg = self.bridge.cv2_to_imgmsg(image, encoding="passthrough")
        self.visualization_publisher.publish(vis_msg)
        
        vis_msg_compressed = CompressedImage()
        vis_msg_compressed.header.stamp = vis_msg.header.stamp
        vis_msg_compressed.format = "jpeg"
        vis_msg_compressed.data = np.array(cv2.imencode('.jpg', image)[1]).tobytes()
        self.visualization_publisher_compressed.publish(vis_msg_compressed)


    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

    def show_masks(self, image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                # boxes
                self.show_box(box_coords, plt.gca())
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    def predict(self, image, input_point, input_label, input_box=None, multimask_output=True):
        # image = Image.open(image_path)
        # image = np.array(image.convert("RGB"))

        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=multimask_output,
        )

        return masks, scores, logits

    def image_callback(self, msg):
        pass

    def detection_callback(self, msg):
        self.detection = msg
        if self.detection is not None and len(self.detection.detections) > 0:
            image = self.detection.detections[0].source_img
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            
            input_point = np.array([[self.detection.detections[0].bbox.center.x, self.detection.detections[0].bbox.center.y]])
            input_box = np.array([self.detection.detections[0].bbox.center.x - self.detection.detections[0].bbox.size_x / 2,
                                self.detection.detections[0].bbox.center.y - self.detection.detections[0].bbox.size_y / 2,
                                self.detection.detections[0].bbox.center.x + self.detection.detections[0].bbox.size_x / 2,
                                self.detection.detections[0].bbox.center.y + self.detection.detections[0].bbox.size_y / 2])
            input_label = np.array([1])
            masks, scores, logits = self.predict(cv_image, input_point, input_label, input_box)
            result_image = self.draw_masks(cv_image, masks, scores, point_coords=input_point, box_coords=input_box, input_labels=input_label)
            
            self.publish_image(result_image)
if __name__ == "__main__":
    rospy.init_node("image_predictor_drone")
    rospy.loginfo("Image predictor node started")
    image_predictor = ImagePredictor("/home/usrg/source/segment-anything-2/checkpoints/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml")
    image_predictor.load_model()
    while not rospy.is_shutdown() and image_predictor.sam2_model is not None:
        rospy.spin()
        
    

        # # Example usage:
        # input_point = np.array([[500, 375]])
        # input_label = np.array([1])
        # input_box = np.array([425, 600, 700, 875])

        # masks, scores, logits = image_predictor.predict('images/truck.jpg', input_point, input_label, input_box)
        # image_predictor.show_masks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)

# Example usage:
# image_predictor = ImagePredictor("../checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml")
# image_predictor.load_model()

# input_point = np.array([[500, 375]])
# input_label = np.array([1])
# input_box = np.array([425, 600, 700, 875])

# masks, scores, logits = image_predictor.predict('images/truck.jpg', input_point, input_label, input_box)
# image_predictor.show_masks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)
