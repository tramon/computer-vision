import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageFilterUtils:
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

    def get_image(self):
        return self.image

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

    def show_images(self, original_title="original image", processed_title="current image"):
        if self.original_image is None or self.image is None:
            raise ValueError(self.image_not_found_error)

        plt.figure(figsize=(16, 8))

        # Оригінальне зображення
        plt.subplot(1, 2, 1)
        plt.title(original_title)
        if len(self.original_image.shape) == 2:  # якщо чорно-біле
            plt.imshow(self.original_image, cmap='gray')
        else:
            plt.imshow(self.original_image)
        plt.axis('off')

        # Оброблене зображення
        plt.subplot(1, 2, 2)
        plt.title(processed_title)
        if len(self.image.shape) == 2:
            plt.imshow(self.image, cmap='gray')
        else:
            plt.imshow(self.image)
        plt.axis('off')

        plt.tight_layout()
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
        :param method: filter name ('gaussian', 'median', 'bilateral', 'adaptive_threshold',
        'morphological_gradient', 'dilation', 'canny')
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

        elif method == "dilation":
            kernel_size = kwargs.get('kernel_size', (5, 5))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            filtered = cv2.dilate(working_image, kernel, iterations=kwargs.get('iterations', 1))

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

        elif method == "sharpen":
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            filtered = cv2.filter2D(working_image, -1, kernel)

        else:
            raise ValueError(f"Filter method '{method}' is not implemented or absent")

        self.image = filtered
        print(f"Filter '{method}' applied")

    def close_open_contours(self, min_distance=5, max_distance=50):
        """
        in some cases - shadows make not-closed contours (that will not be detected as shape)
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

    def draw_contours_by_mask(self, mask, contour_color=(0, 255, 0), thickness=2):
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = self.image.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            cv2.drawContours(output_image, [cnt], -1, contour_color, thickness)

        self.image = output_image
        print(f"[INFO] Objects drawn: {len(contours)}")

    def change_color_saturation(self, lower_bound, upper_bound, factor=1.5):
        """
        changes color saturation

        :param lower_bound: in HSV (example (0, 100, 100))
        :param upper_bound: in HSV (example (10, 255, 255))
        :param factor: lower than one - decreases, bigger than one increases
        """
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        # to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)

        # create mask for a given color
        mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))

        # select only given colors
        h, s, v = cv2.split(hsv)

        # change saturation in the area
        s = np.where(mask > 0, np.clip(s.astype(np.float32) * factor, 0, 255), s)

        # merge
        hsv_modified = cv2.merge([h, s.astype(np.uint8), v])
        modified_rgb = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB)

        self.image = modified_rgb
        print(f"Saturation changed by factor {factor} for selected color range.")

    def show_hue_histogram_comparison(self, channel=0):
        if self.original_image is None or self.image is None:
            raise ValueError(self.image_not_found_error)

        hsv_original = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
        hsv_processed = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)

        hist_h_original = cv2.calcHist([hsv_original], [channel], None, [180], [0, 180])
        hist_h_processed = cv2.calcHist([hsv_processed], [channel], None, [180], [0, 180])

        fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})

        ax[0].plot(np.arange(180), hist_h_original.ravel(), label="Original Image", linestyle='-', color='grey', linewidth=2)
        ax[0].plot(np.arange(180), hist_h_processed.ravel(), label="Processed Image", linestyle='--', color='grey', linewidth=1)
        ax[0].set_title("Hue Histogram Comparison (HSV)")
        ax[0].set_xlabel("Hue value (0-179)")
        ax[0].set_ylabel("Pixel count")
        ax[0].set_xlim(0, 179)
        ax[0].legend()
        ax[0].grid(True)

        hue_bar = np.zeros((20, 180, 3), dtype=np.uint8)
        for i in range(180):
            hue_bar[:, i] = (i, 255, 255)
        hue_bar_bgr = cv2.cvtColor(hue_bar, cv2.COLOR_HSV2RGB)

        ax[1].imshow(hue_bar_bgr, aspect='auto')
        ax[1].set_xticks(np.arange(0, 180, 30))
        ax[1].set_yticks([])
        ax[1].set_xlabel("Hue (HSV) 0-179")

        plt.tight_layout()
        plt.show()

    def find_and_draw_colored_objects(self, base_color_hsv, tolerance=10, box_color=(0, 255, 0), thickness=3):
        """
        finds objects with given mask and contours them

        :param base_color_hsv: base color in HSV in example (120, 255, 255) — blue
        :param tolerance: what deviation of a color is accepted
        :param box_color: a color of the rectangle box that is a contour
        :param thickness: line of th contour thickness
        """
        if self.image is None:
            raise ValueError(self.image_not_found_error)

        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        h_base, s_base, v_base = base_color_hsv

        lower = (max(h_base - tolerance, 0), 50, 50)
        upper = (min(h_base + tolerance, 179), 255, 255)

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = self.image.copy()

        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, thickness)

        self.image = output_image
        print(f"[INFO] Objects found: {len(contours)}")