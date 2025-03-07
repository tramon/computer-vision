import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageFilters:

    def __init__(self):
        self.no_image_error_message = 'No image'

    @staticmethod
    def get_image(image_path):
        image = cv2.imread(image_path)
        image_tuned = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_tuned

    @staticmethod
    def show_image(image, image_explanation_text):
        plt.figure(figsize=(15, 10))
        plt.imshow(image, cmap="gray")
        plt.title(image_explanation_text)
        plt.axis("off")
        plt.show()

    @staticmethod
    def colored_to_monochrome(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def apply_gaussian_blur(self, image, kernel_size=(15, 15)):
        if image is not None:
            blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
            return blurred_image
        else:
            print(self.no_image_error_message)
            return None

    def add_gaussian_noise(self, image, mean=0, sigma=25):
        if image is not None:
            noise = np.random.normal(mean, sigma, image.shape).astype(np.int16)
            gaussian_noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return gaussian_noisy_image
        else:
            print(self.no_image_error_message)
            return None

    def add_salt_pepper_noise(self, image, amount=0.05, salt_vs_pepper=0.5):
        if image is not None:
            noisy_image = image.copy()
            total_pixels = noisy_image.size

            num_salt = int(np.ceil(amount * total_pixels * salt_vs_pepper))
            num_pepper = int(np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)))

            coordinates = [np.random.randint(0, i, num_salt) for i in noisy_image.shape[:2]]
            if len(noisy_image.shape) == 2:
                noisy_image[coordinates[0], coordinates[1]] = 255
            else:
                noisy_image[coordinates[0], coordinates[1], :] = 255

            coordinates = [np.random.randint(0, i, num_pepper) for i in noisy_image.shape[:2]]
            if len(noisy_image.shape) == 2:
                noisy_image[coordinates[0], coordinates[1]] = 0
            else:
                noisy_image[coordinates[0], coordinates[1], :] = 0

            return noisy_image
        else:
            print(self.no_image_error_message)
            return None

    def apply_median_blur(self, image, kernel_size=5):
        if image is not None:
            blurred_image = cv2.medianBlur(image, kernel_size)
            return blurred_image
        else:
            print(self.no_image_error_message)
            return None

    def apply_sharpening(self, image):
        if image is not None:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            sharpened_image = cv2.filter2D(image, -1, kernel)
            return sharpened_image
        else:
            print(self.no_image_error_message)
            return None

    def apply_edge_detection(self, image, threshold1=100, threshold2=200):
        if image is not None:
            edges = cv2.Canny(image, threshold1, threshold2)
            return edges
        else:
            print(self.no_image_error_message)
            return None

    def invert_colors(self, image):
        if image is not None:
            inverted_image = cv2.bitwise_not(image)
            return inverted_image
        else:
            print(self.no_image_error_message)
            return None

    def apply_sepia(self, image):
        if image is not None:
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_image = cv2.transform(image, sepia_filter)
            sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
            return sepia_image
        else:
            print(self.no_image_error_message)
            return None

    def decrease_brightness(self, image, factor=0.5):
        if image is not None:
            bright_image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            return bright_image
        else:
            print(self.no_image_error_message)
            return None

    def increase_brightness(self, image, factor=1.5):
        if image is not None:
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            l = np.clip(l.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            bright_image = cv2.merge((l, a, b))
            bright_image = cv2.cvtColor(bright_image, cv2.COLOR_LAB2BGR)
            return bright_image
        else:
            print(self.no_image_error_message)
            return None

    def detect_objects(self, image, amount_of_objects):
        if image is not None:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_black = np.array([0, 0, 0])
            upper_black = np.array([255, 255, 35])

            darker_objects_mask = cv2.inRange(hsv_image, lower_black, upper_black)

            kernel = np.ones((5, 5), np.uint8)
            darker_objects_mask = cv2.morphologyEx(darker_objects_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(darker_objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
            filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:amount_of_objects]

            contour_image_final = image.copy()
            result = cv2.drawContours(contour_image_final, filtered_contours, -1, (0, 255, 0), 3)

            return result
        else:
            print(self.no_image_error_message)
            return None


if __name__ == '__main__':
    image_filters = ImageFilters()

    skydive = image_filters.get_image('skydive.jpg')
    wind_tunel = image_filters.get_image('wind-tunel.jpg')
    duo_skydive = image_filters.get_image('duo_skydive.jpg')

    skydive_monochrome = image_filters.colored_to_monochrome(wind_tunel)
    image_filters.show_image(skydive_monochrome, 'Monochrome')

    skydive_edges = image_filters.apply_edge_detection(skydive)
    image_filters.show_image(skydive_edges, 'Edge detection')

    blurred_tunnel = image_filters.apply_gaussian_blur(wind_tunel)
    image_filters.show_image(blurred_tunnel, 'Blurred via gaussian method')

    median_blured_skydive = image_filters.apply_median_blur(skydive)
    image_filters.show_image(median_blured_skydive, 'Blurred via median method')

    sharpened_skydive = image_filters.apply_sharpening(skydive)
    image_filters.show_image(sharpened_skydive, 'Sharpened')

    inverted_wind_tunel = image_filters.invert_colors(wind_tunel)
    image_filters.show_image(inverted_wind_tunel, 'Inverted')

    sepia_skydive = image_filters.apply_sepia(skydive)
    image_filters.show_image(sepia_skydive, 'Sepia')

    darker_wind_tunel = image_filters.decrease_brightness(wind_tunel, factor=0.4)
    image_filters.show_image(darker_wind_tunel, 'Darker')

    brighter_skydive2 = image_filters.increase_brightness(wind_tunel, factor=1.5)
    image_filters.show_image(brighter_skydive2, 'Brighter')

    noisy_gaussian_wind_tunel = image_filters.add_gaussian_noise(wind_tunel, sigma=30)
    image_filters.show_image(noisy_gaussian_wind_tunel, 'Gaussian noise')

    salty_noisy_skydive = image_filters.add_salt_pepper_noise(skydive, amount=0.05, salt_vs_pepper=0.5)
    image_filters.show_image(salty_noisy_skydive, 'Salt-pepper noise')

    borders_image = image_filters.detect_objects(wind_tunel, 2)
    image_filters.show_image(borders_image, 'Border detection of two wind-tunel flyers')
