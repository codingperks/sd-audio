from PIL import Image


class preprocessor:
    def crop_and_resize(image_path, output_path, size):
        # Open the image
        image = Image.open(image_path)

        # Crop the image to a square aspect ratio
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2
        cropped_image = image.crop((left, top, right, bottom))

        # Resize the image to the desired size
        resized_image = cropped_image.resize((size, size))

        # Save the resized image
        resized_image.save(output_path)

    def rotate_image(image_path, output_path, angle):
        # Open image
        image = Image.open(image_path)

        # Rotate by specified angle
        rotated_image = image.rotate(angle)

        # Save
        rotated_image.save(output_path)
