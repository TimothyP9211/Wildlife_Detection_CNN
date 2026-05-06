from PIL import Image, ImageDraw, ImageFont



# Draw predicted bounding boxes and labels on the image
class DrawModule():
    def __init__(self, font_size=32):
        self.class_to_names = {0:"Deer", 1:"Moose", 2:"Bear", 3:"Fox", 4:"Wolf", 5:"Bison", 6:"Squirrel"}
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            self.font = ImageFont.load_default()

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
            animal_name = self.class_to_names[int(label.item())]
            text = f"{animal_name} {score.item():.2f}"
            draw.rectangle([x, y, x + w, y + h],outline="red",width=4)
            draw.text((x, y),text,fill="red",font=self.font)

        image.save(output_path)
        print(f"Saved prediction to {output_path}")