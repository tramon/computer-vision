import cv2
import numpy as np
import sys


class TrafficDynamicAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.video_not_found_error = f"Video: '{self.video_path}' was not found"

        self.car_objects = {}  # {id: (cx, cy)}
        self.next_car_id = 0

    def load_video(self):
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise ValueError(self.video_not_found_error)
        print(f"Video '{self.video_path}' was loaded successfully.")

    def adjust_brightness(self, frame, factor):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_filter(self, frame, method="gaussian", **kwargs):
        filtered = frame.copy()
        is_gray_required = method in ["adaptive_threshold", "canny", "morphological_gradient"]

        if is_gray_required:
            if len(filtered.shape) == 3 and filtered.shape[2] == 3:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        if method == "gaussian":
            kernel_size = kwargs.get('kernel_size', (5, 5))
            filtered = cv2.GaussianBlur(filtered, kernel_size, 0)

        elif method == "median":
            kernel_size = kwargs.get('kernel_size', 5)
            filtered = cv2.medianBlur(filtered, kernel_size)

        elif method == "bilateral":
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            filtered = cv2.bilateralFilter(filtered, d, sigma_color, sigma_space)

        elif method == "morphological_gradient":
            kernel_size = kwargs.get('kernel_size', (5, 5))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)

        elif method == "adaptive_threshold":
            block_size = kwargs.get('block_size', 21)
            C = kwargs.get('C', 10)
            filtered = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C)
        elif method == "canny":
            threshold1 = kwargs.get('threshold1', 50)
            threshold2 = kwargs.get('threshold2', 150)
            blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
            filtered = cv2.Canny(blurred, threshold1, threshold2)

        else:
            raise ValueError(f"Filter method '{method}' is not implemented or absent.")

        return filtered

    def draw_tracked_objects(self, frame):
        fg_mask = self.background_subtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2
                area = cv2.contourArea(cnt)
                current_centroids.append((center_x, center_y, x, y, w, h, area))

        updated_objects = {}

        for centroid_x, centroid_y, x, y, w, h, area in current_centroids:
            min_distance = float('inf')
            matched_id = None
            for object_id, (prev_x, prev_y) in self.car_objects.items():
                distance = np.linalg.norm(np.array((centroid_x, centroid_y)) - np.array((prev_x, prev_y)))
                if distance < min_distance and distance < 50:
                    min_distance = distance
                    matched_id = object_id

            if matched_id is None:
                self.next_car_id += 1
                matched_id = self.next_car_id

            updated_objects[matched_id] = (centroid_x, centroid_y)

            # Визначення типу об'єкта за площею
            if area > 3500:
                label_prefix = "Truck"
            else:
                label_prefix = "Car"

            label = f"{label_prefix}-{matched_id}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.car_objects = updated_objects
        return frame

    def draw_detected_objects(self, frame, clean_frame):
        fg_mask = self.background_subtractor.apply(clean_frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2
                current_centroids.append((center_x, center_y, x, y, w, h))

        updated_objects = {}

        for centroid_x, centroid_y, x, y, w, h in current_centroids:
            min_distance = float('inf')
            matched_id = None
            for object_id, (prev_x, prev_y) in self.car_objects.items():
                distance = np.linalg.norm(np.array((centroid_x, centroid_y)) - np.array((prev_x, prev_y)))
                if distance < min_distance and distance < 50:
                    min_distance = distance
                    matched_id = object_id

            if matched_id is None:
                self.next_car_id += 1
                matched_id = self.next_car_id

            updated_objects[matched_id] = (centroid_x, centroid_y)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            area = w * h
            label_type = "Truck" if area > 3500 else "Car"
            label = f"{label_type}-{matched_id}"

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.car_objects = updated_objects

        return frame

    def display_video(self, filters_sequence=None):
        if self.video is None:
            raise ValueError(self.video_not_found_error)

        print("Starting video playback...")

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                print("End of video reached or cannot read the frame.")
                break

            clean_frame = frame.copy()

            frame = self.adjust_brightness(frame, 0.7)
            frame = self.adjust_brightness(frame, 1.1)

            if filters_sequence:
                for filter_config in filters_sequence:
                    method = filter_config.get("method")
                    params = {k: v for k, v in filter_config.items() if k != "method"}
                    frame = self.apply_filter(frame, method=method, **params)

            frame = self.draw_tracked_objects(clean_frame)
            #frame = self.draw_detected_objects(frame, clean_frame)

            cv2.imshow('Traffic Dynamic Analyzer', frame)

            key = cv2.waitKey(30) & 0xFF
            try:
                if cv2.getWindowProperty('Traffic Dynamic Analyzer', cv2.WND_PROP_AUTOSIZE) < 0:
                    print("Video window was closed by the user.")
                    break
            except cv2.error:
                print("Video window was closed (Null Pointer exception caught).")
                break

            if key == ord('q'):
                print("Pressed 'q'. Exiting...")
                break

        self.video.release()
        cv2.destroyAllWindows()
        print("Video and window resources released. Program finished.")
        sys.exit(0)

    def display_video_raw(self, filters_sequence=None):
        if self.video is None:
            raise ValueError(self.video_not_found_error)

        print("Starting RAW video playback...")

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                print("End of video reached or cannot read the frame.")
                break

            clean_frame = frame.copy()

            frame = self.adjust_brightness(frame, 0.7)
            frame = self.adjust_brightness(frame, 1.1)

            if filters_sequence:
                for filter_config in filters_sequence:
                    method = filter_config.get("method")
                    params = {k: v for k, v in filter_config.items() if k != "method"}
                    frame = self.apply_filter(frame, method=method, **params)

            # After filters: ensure frame is color to allow drawing colored boxes
            if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = self.draw_detected_objects(frame, clean_frame)

            cv2.imshow('Traffic Dynamic Analyzer (RAW)', frame)

            key = cv2.waitKey(30) & 0xFF
            try:
                if cv2.getWindowProperty('Traffic Dynamic Analyzer (RAW)', cv2.WND_PROP_AUTOSIZE) < 0:
                    print("Video window was closed by the user.")
                    break
            except cv2.error:
                print("Video window was closed (Null Pointer exception caught).")
                break

        self.video.release()
        cv2.destroyAllWindows()
        print("Video and window resources released. Program finished.")
        sys.exit(0)


if __name__ == "__main__":
    analyzer = TrafficDynamicAnalyzer("traffic.mp4")
    analyzer.load_video()

    filters_sequence = [
        {"method": "gaussian", "kernel_size": (9, 9)},
        {"method": "bilateral", "d": 9, "sigma_color": 90, "sigma_space": 170},
        {"method": "morphological_gradient", "kernel_size": (5, 5)},
        {"method": "canny", "threshold1": 35, "threshold2": 135},
    ]

    analyzer.display_video(filters_sequence=filters_sequence)
