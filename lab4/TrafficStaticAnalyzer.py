import cv2
import matplotlib.pyplot as plt
import numpy as np
from ImageFilterUtils import ImageFilterUtils


class TrafficStaticAnalyzer:
    def __init__(self, image_path):
        self.utils = ImageFilterUtils(image_path)
        self.image_path = image_path

    def load_image(self):
        self.utils.load_image()

    def get_image(self):
        return self.utils.get_image()

    def show_image(self, title="Image"):
        self.utils.show_image(title=title)

    def show_images(self, original_title="Оригінальне зображення", processed_title="Поточне зображення"):
        self.utils.show_images(original_title=original_title, processed_title=processed_title)

    def adjust_brightness(self, factor=0.8):
        self.utils.adjust_brightness(factor=factor)

    def apply_filter(self, method="gaussian", **kwargs):
        self.utils.apply_filter(method=method, **kwargs)

    def close_open_contours(self, min_distance=5, max_distance=50):
        self.utils.close_open_contours(min_distance=min_distance, max_distance=max_distance)

    def draw_contours_by_mask(self, mask, contour_color=(0, 255, 0), thickness=2):
        self.utils.draw_contours_by_mask(mask, contour_color, thickness)

    def reduce_saturation_by_mask(self, mask, factor=0.1):
        self.utils.change_color_saturation(mask, factor)

    def _is_rectangle_overlaps(self, rect1, rect2):
        x11, y11 = rect1[:, 0].min(), rect1[:, 1].min()
        x12, y12 = rect1[:, 0].max(), rect1[:, 1].max()

        x21, y21 = rect2[:, 0].min(), rect2[:, 1].min()
        x22, y22 = rect2[:, 0].max(), rect2[:, 1].max()

        if x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11:
            return False
        else:
            return True

    def _draw_non_intersecting_contours(self, drawn_image, selected_candidates):
        non_intersected_candidates = []
        for candidate in selected_candidates:
            keep = True
            for selected in non_intersected_candidates:
                if self._is_rectangle_overlaps(candidate, selected):
                    keep = False
                    break
            if keep:
                non_intersected_candidates.append(candidate)

        for idx, box in enumerate(non_intersected_candidates, start=1):
            cv2.drawContours(drawn_image, [box], 0, (0, 255, 0), 3)
            top_point = box[np.argmin(box[:, 1])]
            text_x = int(top_point[0]) - 20
            text_y = int(top_point[1]) - 10

            if text_y < 10:
                center_x = int(np.mean(box[:, 0]))
                center_y = int(np.mean(box[:, 1]))
                text_x = center_x - 30
                text_y = center_y

            cv2.putText(drawn_image,
                        f"Truck-{idx}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

        self.utils.image = drawn_image
        print(f"Identified Trucks: {len(non_intersected_candidates)}")

    def _find_trucks(self, current_image, draw_all_contours, min_area, max_area, aspect_ratio_range):
        contours, _ = cv2.findContours(self.utils.image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Contours found: {len(contours)}")

        selected_candidates = []

        if draw_all_contours:
            cv2.drawContours(current_image, contours, -1, (0, 0, 255), 2)

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

            if 4 <= len(approx) <= 8:
                rect = cv2.minAreaRect(approx)
                (center_x, center_y), (width, height), angle = rect

                if width == 0 or height == 0:
                    continue

                aspect_ratio = max(width, height) / min(width, height)
                area = width * height

                if min_area <= area <= max_area and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    selected_candidates.append(box)

        print(f"Trucks found based on input params: {len(selected_candidates)}")

        self._draw_non_intersecting_contours(current_image, selected_candidates)
        self.utils.image = current_image

    def find_truck_raw(self, min_area=3000, max_area=5000, aspect_ratio_range=(2.0, 3.5)):
        if self.utils.image is None:
            raise ValueError(self.utils.image_not_found_error)

        if len(self.utils.image.shape) == 2:
            current_image = cv2.cvtColor(self.utils.image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            current_image = self.utils.image.copy()

        self._find_trucks(
            current_image=current_image,
            draw_all_contours=True,
            min_area=min_area,
            max_area=max_area,
            aspect_ratio_range=aspect_ratio_range
        )

    def find_truck(self, min_area=2000, max_area=5000, aspect_ratio_range=(1.8, 3.5)):
        if self.utils.image is None or self.utils.original_image is None:
            raise ValueError(self.utils.image_not_found_error)

        current_image = self.utils.original_image.copy()

        self._find_trucks(
            current_image=current_image,
            draw_all_contours=False,
            min_area=min_area,
            max_area=max_area,
            aspect_ratio_range=aspect_ratio_range
        )


if __name__ == '__main__':
    # analyzer = TrafficStaticAnalyzer("traffic1.jpg")
    # analyzer = TrafficStaticAnalyzer("traffic2.jpg")
    analyzer = TrafficStaticAnalyzer("traffic3.jpg")
    # analyzer = TrafficStaticAnalyzer("traffic4.jpg")
    analyzer.load_image()
    #analyzer.show_image(title="Original Image")

    analyzer.adjust_brightness(factor=0.7)  # 0.7 - for trucks
    #analyzer.show_image(title="Brightness Optimized to 0.7")
    analyzer.adjust_brightness(factor=1.1)  # 1.1 - for trucks
    #analyzer.show_image(title="Brightness Optimized to 1.1 after 0.7")

    analyzer.apply_filter(method="gaussian", kernel_size=(9, 9))
    #analyzer.show_image(title="Gaussian Blurred")

    analyzer.apply_filter(method="morphological_gradient", kernel_size=(5, 5))
    #analyzer.show_image(title="Morphological gradient")

    analyzer.apply_filter(method="canny", threshold1=35, threshold2=135)
    #analyzer.show_image(title="Canny Edge Detection")

    analyzer.close_open_contours(min_distance=15, max_distance=35)
    #analyzer.show_image(title="Closed Open contours")

    # ------------------------
    # only one of the following methods at a time can be used: find_truck_raw() or find_truck()
    # analyzer.find_truck_raw(min_area=1900, max_area=3000, aspect_ratio_range=(2.0, 3.8))
    analyzer.find_truck(min_area=1900, max_area=3000, aspect_ratio_range=(2.0, 3.8))

    analyzer.show_image(title="Detected Trucks")
