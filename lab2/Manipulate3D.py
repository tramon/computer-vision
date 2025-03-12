from lab1.graphics import *
import numpy as np
import math


class Manipulate3D:

    def __init__(self):
        self.indexes_of_tetrahedron_faces = [
            (0, 1, 2),  # left
            (0, 1, 3),  # bottom
            (1, 2, 3),  # right
            (2, 0, 3)  # back
        ]

    @staticmethod
    def prepare_tetrahedron_points(tetrahedron_size):
        sqrt_3 = np.sqrt(3)
        sqrt_6 = np.sqrt(6)
        points = np.array([
            [0, 0, 0],
            [tetrahedron_size, 0, 0],
            [tetrahedron_size / 2, (tetrahedron_size * sqrt_3) / 2, 0],
            [tetrahedron_size / 2, (tetrahedron_size * sqrt_3) / 6, (tetrahedron_size * sqrt_6) / 3]
        ])

        centroid = np.mean(points, axis=0)
        point_adjusted_to_real_center = points + np.array([0 - centroid[0], 0 - centroid[1], 0])

        return point_adjusted_to_real_center

    @staticmethod
    def convert_coordinates_to_points(points, center_x, center_y):
        windows_points = []
        for p in points:
            windows_points.append(Point(p[0] + center_x, p[1] + center_y))
        return windows_points

    @staticmethod
    def axis_matrix_operations(points, angle, tetrahedron_center, axis='y'):
        rad = math.radians(angle)
        if axis == 'x':
            adjusted_points = points - tetrahedron_center
            matrix = np.array([
                [1, 0, 0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad), math.cos(rad)]
            ])
            rotated_points = np.dot(adjusted_points, matrix.T) + tetrahedron_center
        elif axis == 'y':
            adjusted_points = points - tetrahedron_center
            matrix = np.array([
                [math.cos(rad), 0, math.sin(rad)],
                [0, 1, 0],
                [-math.sin(rad), 0, math.cos(rad)]
            ])
            rotated_points = np.dot(adjusted_points, matrix.T) + tetrahedron_center
        elif axis == 'z':
            rad = math.radians(angle)
            adjusted_points = points - tetrahedron_center
            matrix = np.array([
                [math.cos(rad), -math.sin(rad), 0],
                [math.sin(rad), math.cos(rad), 0],
                [0, 0, 1]
            ])
            rotated_points = np.dot(adjusted_points, matrix.T) + tetrahedron_center
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
        return rotated_points

    def draw_faces(self, window, window_points, colors):
        face_objects = []

        for i, face in enumerate(self.indexes_of_tetrahedron_faces):
            tetrahedron_face = Polygon(window_points[face[0]], window_points[face[1]], window_points[face[2]])
            tetrahedron_face.setFill(colors[i])
            tetrahedron_face.setOutline("black")
            tetrahedron_face.setWidth(3)
            tetrahedron_face.draw(window)
            face_objects.append(tetrahedron_face)
        return face_objects

    @staticmethod
    def undraw_faces(face_objects):
        for face in face_objects:
            face.undraw()

    def draw_tetrahedron(self, win, center_x, center_y, size, colors):
        points = self.prepare_tetrahedron_points(size)
        windows_points = self.convert_coordinates_to_points(points, center_x, center_y)
        face_objects = self.draw_faces(win, windows_points, colors)

        win.getMouse()
        Manipulate3D.undraw_faces(face_objects)

    @staticmethod
    def rotate(points, angle, axis):
        tetrahedron_center = np.mean(points)
        rotated_points = Manipulate3D.axis_matrix_operations(points, angle, tetrahedron_center, axis)

        return rotated_points

    @staticmethod
    def rotate_fixed_vertex(points, angle, axis):
        tetrahedron_center = np.mean(points[:3], axis=0)
        rotated_points = Manipulate3D.axis_matrix_operations(points, angle, tetrahedron_center, axis)

        return rotated_points

    def animate_rotation(self, win, center_x, center_y, size, angle, colors, axis):
        points = Manipulate3D.prepare_tetrahedron_points(size)
        rotated_points = Manipulate3D.rotate(points, angle, axis)
        windows_points = self.convert_coordinates_to_points(rotated_points, center_x, center_y)
        face_objects = self.draw_faces(win, windows_points, colors)
        return face_objects

    def animate_rotation_fixed_vertex(self, win, center_x, center_y, size, angle, colors, axis):
        points = Manipulate3D.prepare_tetrahedron_points(size)
        rotated_points = Manipulate3D.rotate_fixed_vertex(points, angle, axis)
        windows_points = self.convert_coordinates_to_points(rotated_points, center_x, center_y)
        face_objects = self.draw_faces(win, windows_points, colors)
        return face_objects

    def infinite_animation(self, window, center_x, center_y, size, colors, axis):
        angle = 0
        while True and window.checkMouse() is None:
            tetrahedron = self.animate_rotation(window, center_x, center_y, size, angle, colors, axis)
            time.sleep(0.05)
            for face in tetrahedron:
                face.undraw()

            angle += 2

    def infinite_animation_fixed_vertex(self, window, center_x, center_y, size, colors, axis):
        angle = 0
        while True and window.checkMouse() is None:
            tetrahedron = self.animate_rotation_fixed_vertex(window, center_x, center_y, size, angle, colors, axis)
            time.sleep(0.05)
            for face in tetrahedron:
                face.undraw()

            angle += 2


if __name__ == "__main__":
    window = GraphWin("Task - 2: 3D projection of Tetrahedron rotations", 1000, 1000)
    window.setCoords(0, 0, 1000, 1000)
    window.setBackground('white')

    manipulate3d = Manipulate3D()

    center_x = 500
    center_y = 500
    size = 350
    colors = ['', 'orange', 'cyan', '']

    manipulate3d.draw_tetrahedron(window, center_x, center_y, size, colors)

    manipulate3d.infinite_animation(window, center_x, center_y, size, colors, axis='x')

    manipulate3d.infinite_animation_fixed_vertex(window, center_x, center_y, size, colors, axis='z')
