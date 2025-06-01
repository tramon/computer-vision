import cv2
import matplotlib.pyplot as plt
from lab4.ImageFilterUtils import ImageFilterUtils


class ImageComparator:
    def __init__(self, image_path_1, image_path_2):
        self.utils1 = ImageFilterUtils(image_path_1)
        self.utils2 = ImageFilterUtils(image_path_2)
        self.image1 = None
        self.image2 = None

    def preprocess_for_comparison(self, filters_img1: list, filters_img2: list):
        def apply_pipeline(utils: ImageFilterUtils, pipeline: list):
            utils.load_image()  # завжди починаємо з оригіналу

            for step in pipeline:
                method = step["method"]
                params = step.get("params", {})
                if method == "saturation":
                    utils.change_color_saturation(**params)
                elif method == "brightness":
                    utils.adjust_brightness(**params)
                elif method == "filter":
                    utils.apply_filter(**params)
                else:
                    print(f"Method not implemented: {method}")

        apply_pipeline(self.utils1, filters_img1)
        apply_pipeline(self.utils2, filters_img2)

        self.image1 = self.utils1.get_image()
        self.image2 = self.utils2.get_image()

        print("Image Preprocessing finished")

    def highlight_identification_objects(self, base_color_hsv=(0, 200, 200), tolerance=10):
        self.utils1.find_and_draw_colored_objects(base_color_hsv=base_color_hsv, tolerance=tolerance)
        self.utils2.find_and_draw_colored_objects(base_color_hsv=base_color_hsv, tolerance=tolerance)
        self.image1 = self.utils1.get_image()
        self.image2 = self.utils2.get_image()
        print("Objects highlighted.")

    def compare_with_orb(self, max_matches_displayed=20, message="ORB Keypoint Matches"):
        if self.image1 is None or self.image2 is None:
            raise ValueError("Images must be loaded")

        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print("One of the images has no descriptors")
            return 0, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        num_matches = len(matches)
        probability = min(1.0, num_matches / 100)

        print(f"Matches found: {num_matches}")
        print(f"object identification probability: {probability:.2f}")

        matched_image = cv2.drawMatches(self.image1, kp1, self.image2, kp2,
                                        matches[:max_matches_displayed], None, flags=2)
        plt.figure(figsize=(16, 8))
        plt.imshow(matched_image)
        plt.title(message)
        plt.axis('off')
        plt.show()

        return num_matches, probability

if __name__ == '__main__':

    comparator0 = ImageComparator("google.jpg", "google2.jpg")
    comparator0.preprocess_for_comparison(filters_img1=[], filters_img2=[])
    comparator0.highlight_identification_objects(base_color_hsv=(0, 200, 200), tolerance=30)
    comparator0.highlight_identification_objects(base_color_hsv=(110, 180, 180), tolerance=1)
    comparator0.compare_with_orb(message='An example of how it should work in perfect scenario - good result')

    filters_img1 = [
        {"method": "saturation",
         "params": {"lower_bound": (0, 120, 120), "upper_bound": (15, 255, 255), "factor": 1.9}},
        {"method": "saturation", "params": {"lower_bound": (80, 90, 0), "upper_bound": (140, 255, 255), "factor": 1.9}},
        {"method": "saturation", "params": {"lower_bound": (30, 0, 0), "upper_bound": (75, 255, 255), "factor": 0.1}},
        {"method": "saturation", "params": {"lower_bound": (5, 10, 20), "upper_bound": (45, 255, 220), "factor": 0.1}},
        {"method": "saturation", "params": {"lower_bound": (2, 50, 0), "upper_bound": (105, 180, 255), "factor": 0.1}},
        {"method": "brightness", "params": {"factor": 0.7}},
        {"method": "filter", "params": {"method": "sharpen"}}
    ]
    filters_img2 = [
        {"method": "saturation",
         "params": {"lower_bound": (0, 120, 120), "upper_bound": (25, 255, 255), "factor": 1.9}},
        {"method": "filter", "params": {"method": "gaussian", "kernel_size": (13, 13)}},
        {"method": "saturation", "params": {"lower_bound": (80, 90, 0), "upper_bound": (140, 255, 255), "factor": 1.9}},
        {"method": "saturation", "params": {"lower_bound": (25, 0, 0), "upper_bound": (100, 255, 255), "factor": 0.0}},
        {"method": "saturation", "params": {"lower_bound": (115, 0, 0), "upper_bound": (255, 255, 255), "factor": 0.0}},
        {"method": "saturation", "params": {"lower_bound": (5, 10, 20), "upper_bound": (45, 255, 220), "factor": 0.0}},
        {"method": "brightness", "params": {"factor": 0.8}}
    ]

    comparator = ImageComparator("dataspace.jpg", "google.jpg")
    comparator.preprocess_for_comparison(filters_img1=filters_img1, filters_img2=filters_img2)
    comparator.highlight_identification_objects(base_color_hsv=(0, 200, 200), tolerance=30)
    comparator.highlight_identification_objects(base_color_hsv=(110, 180, 180), tolerance=1)
    comparator.compare_with_orb(message='This is a real result which is not satisfactory')