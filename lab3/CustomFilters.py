import numpy as np

from term.ImageFilters import ImageFilters


class CustomFiltering:

    def __init__(self):
        self.no_image_error_message = 'No image'
        self.please_wait_message = 'It takes a while for the image to render. \n\tPlease wait...'

    def apply_filter_diagonal_from_top(self, image, filter_function):
        if image is None:
            print(self.no_image_error_message)
            return None

        height, width, channels = image.shape
        new_image = image.copy()

        for y in range(height):
            for x in range(width):
                factor = abs(x / width - y / height)
                new_image[y, x] = filter_function(image[y, x], factor)

        return new_image

    def apply_filter_diagonal_from_bottom(self, image, filter_function):
        if image is None:
            print(self.no_image_error_message)
            return None

        height, width, channels = image.shape
        new_image = image.copy()

        for y in range(height):
            for x in range(width):
                factor = abs(x / width - (height - y) / height)
                new_image[y, x] = filter_function(image[y, x], factor)

        return new_image

    def apply_filter_to_center(self, image, filter_function):
        if image is None:
            print(self.no_image_error_message)
            return None

        height, width, channels = image.shape
        center_x, center_y = width // 2, height // 2
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        new_image = image.copy()

        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                factor = 1 - (distance / max_distance)
                new_image[y, x] = filter_function(image[y, x], factor)

        return new_image

    def apply_filter_from_center(self, image, filter_function):
        if image is None:
            print(self.no_image_error_message)
            return None

        height, width, channels = image.shape
        center_x, center_y = width // 2, height // 2
        max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
        new_image = image.copy()

        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                factor = distance / max_distance
                new_image[y, x] = filter_function(image[y, x], factor)

        return new_image

    @staticmethod
    def brightness_gradient(pixel, factor):
        return np.clip(pixel * (1 - factor), 0, 255).astype(np.uint8)

    @staticmethod
    def sepia_gradient(pixel, factor):
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_pixel = np.dot(sepia_filter, pixel)
        blended_pixel = sepia_pixel * (1 - factor) + pixel * factor
        return np.clip(blended_pixel, 0, 255).astype(np.uint8)

    @staticmethod
    def monochrome_gradient(pixel, factor):
        gray_value = np.dot(pixel, [0.299, 0.587, 0.114])
        blended_pixel = gray_value * (1 - factor) + pixel * factor
        return np.clip(blended_pixel, 0, 255).astype(np.uint8)

    @staticmethod
    def inverted_gradient(pixel, factor):
        inverted_pixel = 255 - pixel
        blended_pixel = inverted_pixel * (1 - factor) + pixel * factor
        return np.clip(blended_pixel, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    custom_filtering = CustomFiltering()
    image_filters = ImageFilters()

    clouds = image_filters.get_image('clouds.jpg')
    image_filters.show_image(clouds, 'Original image')
    print(custom_filtering.please_wait_message)

    diagonal_filtered = custom_filtering.apply_filter_diagonal_from_top(clouds, custom_filtering.brightness_gradient)
    image_filters.show_image(diagonal_filtered, 'Brightness Gradient Diagonal, top left -> bottom right')
    print(custom_filtering.please_wait_message)

    center_filtered = custom_filtering.apply_filter_from_center(clouds, custom_filtering.sepia_gradient)
    image_filters.show_image(center_filtered, 'Sepia Gradient from Center')
    print(custom_filtering.please_wait_message)

    diagonal_filtered = custom_filtering.apply_filter_diagonal_from_bottom(clouds, custom_filtering.monochrome_gradient)
    image_filters.show_image(diagonal_filtered, 'Monochrome Gradient Diagonal, bottom left -> top right')
    print(custom_filtering.please_wait_message)

    center_filtered = custom_filtering.apply_filter_from_center(clouds, custom_filtering.inverted_gradient)
    image_filters.show_image(center_filtered, 'Inverted Gradient from Center')
    print(custom_filtering.please_wait_message)

    center_filtered = custom_filtering.apply_filter_to_center(clouds, custom_filtering.inverted_gradient)
    image_filters.show_image(center_filtered, 'Inverted Gradient to Center')