import math

from lab1.graphics import *


class Manipulate2D:

    @staticmethod
    def prepare_rhombus_points(center_point, figure_size):

        rhombus_points_list = [
            Point(center_point.getX(), center_point.getY() + figure_size),  # top
            Point(center_point.getX() + figure_size, center_point.getY()),  # right
            Point(center_point.getX(), center_point.getY() - figure_size),  # bottom
            Point(center_point.getX() - figure_size, center_point.getY())  # left
        ]
        for p in rhombus_points_list:
            print(f' Point: {int(p.getX())} : {int(p.getY())}  ')

        return rhombus_points_list

    @staticmethod
    def move_rhombus(figure, dx, dy, steps=100, delay=0.02):
        for _ in range(steps):
            figure.move(dx / steps, dy / steps)
            time.sleep(delay)
        return figure

    @staticmethod
    def get_center_point(figure):
        figure_points = figure.getPoints()
        figure_center_x = sum(p.getX() for p in figure_points) / len(figure_points)
        figure_center_y = sum(p.getY() for p in figure_points) / len(figure_points)
        return Point(figure_center_x, figure_center_y)

    @staticmethod
    def rotate_rhombus(window, figure, center_point, angle_degrees, steps=100, delay=0.02):
        angle_radians = math.radians(angle_degrees)
        step_angle = angle_radians / steps  # Small angle increment

        for _ in range(steps):
            new_points = []
            for point in figure.getPoints():
                # Convert coordinates relative to center
                dx = point.getX() - center_point.getX()
                dy = point.getY() - center_point.getY()

                # Apply rotation
                new_x = center_point.getX() + dx * math.cos(step_angle) - dy * math.sin(step_angle)
                new_y = center_point.getY() + dx * math.sin(step_angle) + dy * math.cos(step_angle)

                new_points.append(Point(new_x, new_y))

            fill_color = figure.getFill()
            outline_color = figure.getOutline()
            outline_width = figure.getWidth()

            figure.undraw()
            figure = Polygon(new_points)
            figure.setFill(fill_color)
            figure.setOutline(outline_color)
            figure.setWidth(outline_width)
            figure.draw(window)
            time.sleep(delay)

        return figure

    @staticmethod
    def resize_rhombus(window, figure, center_point, scale_factor, enlarge, steps=100, delay=0.02):
        for _ in range(steps):
            new_points = []
            for point in figure.getPoints():

                # Calculate distances from center
                dx = point.getX() - center_point.getX()
                dy = point.getY() - center_point.getY()

                # Apply scaling
                if enlarge:
                    new_x = center_point.getX() + dx * (1 + scale_factor / steps)
                    new_y = center_point.getY() + dy * (1 + scale_factor / steps)
                elif not enlarge:
                    new_x = center_point.getX() + dx * (1 - scale_factor / steps)
                    new_y = center_point.getY() + dy * (1 - scale_factor / steps)
                else:
                    raise ValueError("The figure can only be enlarged or shrinked")

                new_points.append(Point(new_x, new_y))

            fill_color = figure.getFill()
            outline_color = figure.getOutline()
            outline_width = figure.getWidth()

            figure.undraw()
            figure = Polygon(new_points)
            figure.setFill(fill_color)
            figure.setOutline(outline_color)
            figure.setWidth(outline_width)
            figure.draw(window)
            time.sleep(delay)
        return figure


if __name__ == '__main__':
    # Task-1
    window = GraphWin("Task-1 - Rombus - move, rotate, re-size", 1000, 1000)
    window.setCoords(0, 0, 1000, 1000)

    manipulate2D = Manipulate2D()

    center_x = 500
    center_y = 500
    rombus_size = 200

    points = manipulate2D.prepare_rhombus_points(Point(center_x, center_y), rombus_size)

    rhombus = Polygon(points)
    rhombus.setFill("cyan")
    rhombus.setWidth(2)
    rhombus.setOutline("grey")
    rhombus.draw(window)

    window.getMouse()
    rhombus = manipulate2D.move_rhombus(rhombus, dx=150, dy=-250, steps=50, delay=0.02)
    rhombus = manipulate2D.rotate_rhombus(window, rhombus, Point(center_x, center_y), angle_degrees=90)
    rhombus = manipulate2D.resize_rhombus(window, rhombus, Point(center_x, center_y), enlarge=False, scale_factor=1.5)

    window.close()
