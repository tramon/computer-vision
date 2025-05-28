
from lab1.graphics import *
import numpy as np
import math
import time


class ManipulatePyramid:
    def __init__(self):
        self.indexes_of_faces = [
            (0, 1, 2, 3),
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4)
        ]

    @staticmethod
    def prepare_pyramid_points(base_size, height):
        half = base_size / 2
        base = [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0]
        ]
        apex = [[0, 0, height]]
        return np.array(base + apex)

    @staticmethod
    def convert_coordinates_to_points(points, center_x, center_y):
        windows_points = []
        for p in points:
            windows_points.append(Point(p[0] + center_x, p[1] + center_y))
        return windows_points

    @staticmethod
    def axis_matrix_operations(points, angle, center, axis='y'):
        rad = math.radians(angle)
        adjusted_points = points - center

        if axis == 'x':
            matrix = np.array([
                [1, 0, 0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad), math.cos(rad)]
            ])
        elif axis == 'y':
            matrix = np.array([
                [math.cos(rad), 0, math.sin(rad)],
                [0, 1, 0],
                [-math.sin(rad), 0, math.cos(rad)]
            ])
        elif axis == 'z':
            matrix = np.array([
                [math.cos(rad), -math.sin(rad), 0],
                [math.sin(rad), math.cos(rad), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

        rotated_points = np.dot(adjusted_points, matrix.T) + center
        return rotated_points

    def draw_faces(self, window, window_points, colors):
        face_objects = []
        for i, face in enumerate(self.indexes_of_faces):
            polygon = Polygon(*[window_points[idx] for idx in face])
            polygon.setFill(colors[i] if i < len(colors) else '')
            polygon.setOutline("black")
            polygon.setWidth(2)
            polygon.draw(window)
            face_objects.append(polygon)
        return face_objects

    @staticmethod
    def undraw_faces(face_objects):
        for face in face_objects:
            face.undraw()

    def draw_pyramid(self, win, center_x, center_y, base_size, height, colors):
        points = self.prepare_pyramid_points(base_size, height)
        windows_points = self.convert_coordinates_to_points(points, center_x, center_y)
        face_objects = self.draw_faces(win, windows_points, colors)
        win.getMouse()
        self.undraw_faces(face_objects)

    def animate_rotation(self, window, center_x, center_y, base_size, height, colors, axis):
        angle = 0
        while True and window.checkMouse() is None:
            points = self.prepare_pyramid_points(base_size, height)
            center = np.mean(points, axis=0)
            rotated_points = self.axis_matrix_operations(points, angle, center, axis)
            windows_points = self.convert_coordinates_to_points(rotated_points, center_x, center_y)
            face_objects = self.draw_faces(window, windows_points, colors)
            time.sleep(0.05)
            self.undraw_faces(face_objects)
            angle += 2


if __name__ == "__main__":
    window = GraphWin("3D Pyramid in Axonometric Projection", 1000, 1000)
    window.setCoords(0, 0, 1000, 1000)
    window.setBackground("white")

    center_x = 500
    center_y = 500
    base_size = 300
    height = 300
    colors = ['gray', 'orange', 'cyan', '', 'yellow']

    mp = ManipulatePyramid()
    mp.draw_pyramid(window, center_x, center_y, base_size, height, colors)
    mp.animate_rotation(window, center_x, center_y, base_size, height, colors, axis='x')
    mp.animate_rotation(window, center_x, center_y, base_size, height, colors, axis='y')
    mp.animate_rotation(window, center_x, center_y, base_size, height, colors, axis='z')