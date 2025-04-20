import cv2
import matplotlib.pyplot as plt
import numpy as np


class TrafficStaticAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_not_found_error = f"Image: '{self.image_path}' was not found"
        self.image = None   # working image to apply filters
        self.original_image = None # original image - to show on the final view

    def load_image(self):
        self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise ValueError(self.image_not_found_error)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_image = self.image.copy()
        print(f"Image '{self.image_path}' was loaded successfully")

    def show_image(self, title="Image"):
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        plt.figure(figsize=(15, 10))
        if len(self.image.shape) == 2:  # black and white
            plt.imshow(self.image, cmap="gray")
        else:  # colored
            plt.imshow(self.image)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def adjust_brightness(self, factor=0.8):
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        # convert to LAB
        lab = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_channel = np.clip(l_channel.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # return to RGB
        adjusted_lab = cv2.merge((l_channel, a_channel, b_channel))
        adjusted_rgb = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)

        self.image = adjusted_rgb
        if factor < 1.0:
            print(f"Brightness decreased in {factor} times")
        else:
            print(f"Brightness increased in  {factor} times.")

    def apply_filter(self, method="gaussian", **kwargs):
        """
        applies a chosen filter and saves the result in a self.image.
        :param method: filter name ('gaussian', 'median', 'bilateral', 'adaptive_threshold', 'morphological_gradient', 'canny')
        :param kwargs: additional filter-specific parameters
        """
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        working_image = self.image.copy()

        is_gray_required = method in ["adaptive_threshold", "canny"]
        if is_gray_required:
            if len(working_image.shape) == 3 and working_image.shape[2] == 3:
                working_image = cv2.cvtColor(working_image, cv2.COLOR_RGB2GRAY)

        if method == "gaussian":
            kernel_size = kwargs.get('kernel_size', (5, 5))
            filtered = cv2.GaussianBlur(working_image, kernel_size, 0)

        elif method == "median":
            kernel_size = kwargs.get('kernel_size', 5)
            filtered = cv2.medianBlur(working_image, kernel_size)

        elif method == "bilateral":
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            filtered = cv2.bilateralFilter(working_image, d, sigma_color, sigma_space)

        elif method == "adaptive_threshold":
            block_size = kwargs.get('block_size', 21)
            c = kwargs.get('C', 10)
            filtered = cv2.adaptiveThreshold(
                working_image, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                c
            )

        elif method == "canny":
            threshold1 = kwargs.get('threshold1', 50)
            threshold2 = kwargs.get('threshold2', 150)
            blurred = cv2.GaussianBlur(working_image, (5, 5), 0)
            filtered = cv2.Canny(blurred, threshold1, threshold2)

        elif method == "morphological_gradient":
            kernel_size = kwargs.get('kernel_size', (5, 5))
            if len(working_image.shape) == 3:
                working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            filtered = cv2.morphologyEx(working_image, cv2.MORPH_GRADIENT, kernel)

        else:
            raise ValueError(f"Filter method '{method}' is not implemented or absent")

        self.image = filtered
        print(f"Filter '{method}' applied")

    def close_open_contours(self, min_distance=5, max_distance=50):
        """
        in some cases - shadows make not-closed contours (that will not be detcted as shape)
        this method aims to close nearest neighboring pixels to finish a shape
        """
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        working_image = self.image.copy()

        if len(working_image.shape) == 3:
            working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

        # convert to binary
        _, binary = cv2.threshold(working_image, 50, 255, cv2.THRESH_BINARY)

        # pixel coordinates
        white_pixels = np.argwhere(binary == 255)

        final_points = []
        for (y, x) in white_pixels:
            neighbors = binary[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
            white_neighbors = cv2.countNonZero(neighbors) - 1  # remove itself (white pixel)
            if white_neighbors == 1:
                final_points.append((x, y))

        print(f"Number of final points found: {len(final_points)}")

        # draw merging lines between points if we are in allowed range
        closed_image = binary.copy()
        for i in range(len(final_points)):
            for j in range(i + 1, len(final_points)):
                pt1 = final_points[i]
                pt2 = final_points[j]
                distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
                if min_distance <= distance <= max_distance:
                    cv2.line(closed_image, pt1, pt2, 255, 1)

        self.image = closed_image
        print(f"Drawing merging lines between points")

    @staticmethod
    def _is_rectangle_overlaps(rect1, rect2):
        """
        checks if tro rectangles overlap\intersect
        rect1 and rect2 — are arrays with 4 points (boxPoints).
        """
        x11, y11 = rect1[:, 0].min(), rect1[:, 1].min()
        x12, y12 = rect1[:, 0].max(), rect1[:, 1].max()

        x21, y21 = rect2[:, 0].min(), rect2[:, 1].min()
        x22, y22 = rect2[:, 0].max(), rect2[:, 1].max()

        # if one rectangle is  out of another rectangle (to the top bottom left or right) - than it does not intersect
        if x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11:
            return False
        else:
            return True

    def _draw_non_intersecting_contours(self, drawn_image, selected_candidates):
        """
        in rare cases (image: traffic3.jpg) same contours are duplicated which leads to finding redundant same trucks
        this method removes rectangles (trucks) that overlap\intersect
        uses additional method `_is_rectangle_overlaps`
        Also adds text on top if we have space for that
        Or draws in the middle if we do not have tex t for that
        """

        non_intersected_candidates = []
        for candidate in selected_candidates:
            keep = True
            for selected in non_intersected_candidates:
                if self._is_rectangle_overlaps(candidate, selected):
                    keep = False
                    break
            if keep:
                non_intersected_candidates.append(candidate)

        image_height, image_width = drawn_image.shape[:2]

        for idx, box in enumerate(non_intersected_candidates, start=1):
            cv2.drawContours(drawn_image, [box], 0, (0, 255, 0), 3)

            # find the top point for the text to be appended
            top_point = box[np.argmin(box[:, 1])]
            text_x = int(top_point[0]) - 20
            text_y = int(top_point[1]) - 10  # above the rectangle

            # f we are out of the image from the top we write text in the center
            if text_y < 10:  # If text is too high
                center_x = int(np.mean(box[:, 0]))
                center_y = int(np.mean(box[:, 1]))
                text_x = center_x - 30  # minus 30 pixels to the left to be more in the center
                text_y = center_y

            # write text
            cv2.putText(drawn_image,
                        f"Truck-{idx}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

        self.image = drawn_image
        print(f"Drawn Trucks: {len(non_intersected_candidates)}")

    def _find_trucks(self, current_image, draw_all_contours, min_area, max_area, aspect_ratio_range):
        """
        internal method to find trucks to be reused for raw and final methods.
        - draws all contours if selected
        - finds objects with range of area:  min_area, max_area
        - finds objects with aspect ratio range
        - draws object lines in green
        - Adds text - Truck-id (where id - is a number of truck)
        """

        contours, _ = cv2.findContours(self.image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
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
        self.image = current_image

    def find_truck_raw(self, min_area=3000, max_area=5000, aspect_ratio_range=(2.0, 3.5)):
        """
        method to find and draw raw contours and found trucks for debugging purposes on a filtered image
        """
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        if len(self.image.shape) == 2:
            current_image = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            current_image = self.image.copy()

        self._find_trucks(
            current_image=current_image,
            draw_all_contours=True,
            min_area=min_area,
            max_area=max_area,
            aspect_ratio_range=aspect_ratio_range
        )

    def find_truck(self, min_area=2000, max_area=5000, aspect_ratio_range=(1.8, 3.5)):
        """
        final method to show truck contours on an original image
        """
        if self.image is None or self.original_image is None:
            raise ValueError(self.image_not_found_error)

        current_image = self.original_image.copy()

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
    # analyzer.show_image(title="Original Image")

    analyzer.adjust_brightness(factor=0.7)  # 0.7 - for trucks
    analyzer.adjust_brightness(factor=1.1)  # 1.1 - for trucks
    # analyzer.show_image(title="Brightness Optimized")

    analyzer.apply_filter(method="gaussian", kernel_size=(9, 9))
    # analyzer.show_image(title="Gaussian Blurred")

    analyzer.apply_filter(method="morphological_gradient", kernel_size=(5, 5))
    # analyzer.show_image(title="Morphological gradient")

    analyzer.apply_filter(method="canny", threshold1=35, threshold2=135)
    # analyzer.show_image(title="Canny Edge Detection")

    analyzer.close_open_contours(min_distance=15, max_distance=35)
    # analyzer.show_image(title="Closed Open contours")

    # ------------------------
    # analyzer.find_truck_raw(min_area=1900, max_area=3000, aspect_ratio_range=(2.0, 3.8))
    analyzer.find_truck(min_area=1900, max_area=3000, aspect_ratio_range=(2.0, 3.8))
    analyzer.show_image(title="Detected Trucks")
