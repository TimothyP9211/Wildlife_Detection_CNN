from PIL import Image
from PIL import ImageDraw

class DrawModule():
    def __init__(self):
        pass

    def boxes_to_pixels(self, boxes, image_width, image_height):
        boxes = boxes.clone()
        boxes[:, 0] = boxes[:, 0] * image_width   
        boxes[:, 1] = boxes[:, 1] * image_height   
        boxes[:, 2] = boxes[:, 2] * image_width    
        boxes[:, 3] = boxes[:, 3] * image_height   
        return boxes

    def draw_predictions(self, image_path, boxes, labels, scores, output_path="output.jpg"):
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        pixel_boxes = self.boxes_to_pixels(boxes.cpu(), image_width, image_height)
        draw = ImageDraw.Draw(image)

        for box, label, score in zip(pixel_boxes, labels.cpu(), scores.cpu()):
            x, y, w, h = box.tolist()
            draw.rectangle([x, y, x + w, y + h],outline="red",width=3)
            draw.text((x, y),f"class {label.item()} {score.item():.8f}",fill="red")

        image.save(output_path)
        print(f"Saved prediction to {output_path}")