from lab4.ImageFilterUtils import ImageFilterUtils


class SatObjectFinder:
    def __init__(self, image_path):
        self.utils = ImageFilterUtils(image_path)
        self.image_path = image_path

    def load_image(self):
        self.utils.load_image()

    def show_images(self, original_title="Original image", processed_title="Current image"):
        self.utils.show_images(original_title=original_title, processed_title=processed_title)


if __name__ == "__main__":
    image_path = "dataspace_300m.jpg"
    # image_path = "googlemaps_200m.jpg"
    analyzer = ImageFilterUtils(image_path)
    analyzer.load_image()
    # analyzer.show_image('Original satellite image')

    # Enhance red colors – to simplify future detection
    analyzer.change_color_saturation(
        lower_bound=(0, 120, 120),
        upper_bound=(15, 255, 255),
        factor=1.9
    )

    # Enhance blue – to simplify future detection
    analyzer.change_color_saturation(
        lower_bound=(80, 90, 0),
        upper_bound=(140, 255, 255),
        factor=1.9
    )

    # Remove green – as a background color we are not interested in
    analyzer.change_color_saturation(
        lower_bound=(30, 0, 0),
        upper_bound=(75, 255, 255),
        factor=0.0
    )

    # Remove brown that partially overlaps with red hue range
    analyzer.change_color_saturation(
        lower_bound=(5, 10, 20),
        upper_bound=(45, 255, 220),
        factor=0.0
    )

    # Remove more brown from yellowish tones
    analyzer.change_color_saturation(
        lower_bound=(2, 50, 0),
        upper_bound=(105, 180, 255),
        factor=0.0
    )

    analyzer.adjust_brightness(1.5)
    analyzer.apply_filter(method="gaussian", kernel_size=(5, 5))

    analyzer.show_hue_histogram_comparison()

    analyzer.find_and_draw_colored_objects(base_color_hsv=(0, 200, 200), tolerance=10)
    analyzer.find_and_draw_colored_objects(base_color_hsv=(120, 200, 200), tolerance=15)
    analyzer.show_images()

    # ----------------------------------------------------------------------------

    image_path_google = "googlemaps_200m.jpg"
    analyzer_google = ImageFilterUtils(image_path_google)
    analyzer_google.load_image()

    # Increase red saturation for the future easier identification
    analyzer_google.change_color_saturation(
        lower_bound=(0, 120, 120),
        upper_bound=(25, 255, 255),
        factor=1.9
    )

    analyzer_google.apply_filter(method="gaussian", kernel_size=(5, 5))

    # Increase blue saturation for the future easier identification
    analyzer_google.change_color_saturation(
        lower_bound=(80, 90, 0),
        upper_bound=(140, 255, 255),
        factor=1.9
    )

    # Remove green from the bottom
    analyzer_google.change_color_saturation(
        lower_bound=(25, 0, 0),
        upper_bound=(100, 255, 255),
        factor=0.0
    )

    # Remove green from the top
    analyzer_google.change_color_saturation(
        lower_bound=(115, 0, 0),
        upper_bound=(255, 255, 255),
        factor=0.0
    )

    # Remove brown
    analyzer_google.change_color_saturation(
        lower_bound=(5, 10, 20),
        upper_bound=(45, 255, 220),
        factor=0.0
    )

    analyzer_google.show_hue_histogram_comparison()
    analyzer_google.adjust_brightness(0.8)

    # Find and draw red buildings
    analyzer_google.find_and_draw_colored_objects(base_color_hsv=(0, 200, 200), tolerance=30)

    # Find and draw blue buildings
    analyzer_google.find_and_draw_colored_objects(base_color_hsv=(110, 180, 180), tolerance=1)

    analyzer_google.show_images()