from PIL import Image

def pad_image_to_square(image):
    # pad the image with zeros to make it square
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), (0, 0, 0))
        result.paste(image, (0, int((width - height) / 2)))
        return result
    else:
        result = Image.new(image.mode, (height, height), (0, 0, 0))
        result.paste(image, (int((height - width) / 2), 0))
        return result