from lab1.graphics import *
import numpy as np
import math


class Manipulate3D:

    def __init__(self):
        self.indexes_of_tetrahedron_faces = [
            (0, 1, 2),  # left
            (0, 1, 3),  # bottom
            (1, 2, 3),  # right
            (2, 0, 3)   # back
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
    def rotate_y_fixed_top_center(points, angle):
        rad = math.radians(angle)
        tetrahedron_center = np.mean(points)
        top_vertex_initial = points[2].copy()  # fixate top vertex

        translated = points - tetrahedron_center
        matrix = np.array([
            [math.cos(rad), 0, math.sin(rad)],
            [0, 1, 0],
            [-math.sin(rad), 0, math.cos(rad)]
        ])

        rotated = np.dot(translated, matrix.T) + tetrahedron_center
        rotated[2] = top_vertex_initial  # get fixated top vertex and overwrite

        return rotated

    @staticmethod
    def convert_coordinates_to_points(points, center_x, center_y):
        windows_points = []
        for p in points:
            windows_points.append(Point(p[0] + center_x, p[1] + center_y))
        return windows_points

    @staticmethod
    def undraw_faces(face_objects):
        for face in face_objects:
            face.undraw()

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

    def draw_tetrahedron(self, win, center_x, center_y, size, colors):
        points = self.prepare_tetrahedron_points(size)
        windows_points = self.convert_coordinates_to_points(points, center_x, center_y)

        face_objects = self.draw_faces(win, windows_points, colors)

        win.getMouse()

        Manipulate3D.undraw_faces(face_objects)

    def animate_tetrahedron_rotation(self, win, center_x, center_y, angle, colors):
        points = Manipulate3D.prepare_tetrahedron_points(size)
        rotated_points = Manipulate3D.rotate_y_fixed_top_center(points, angle)

        windows_points = self.convert_coordinates_to_points(rotated_points, center_x, center_y)

        face_objects = self.draw_faces(win, windows_points, colors)

        return face_objects

    def infinite_rotation(self, window, center_x, center_y, colors):
        angle = 0
        while True and window.checkMouse() is None:
            tetrahedron = self.animate_tetrahedron_rotation(window, center_x, center_y, angle, colors)
            time.sleep(0.15)
            for face in tetrahedron:
                face.undraw()

            angle += 5


if __name__ == "__main__":
    window = GraphWin("Task - 2: 3D projection of Tetrahedron", 1000, 1000)
    window.setCoords(0, 0, 1000, 1000)
    window.setBackground('white')

    manipulate3d = Manipulate3D()

    center_x = 500
    center_y = 500
    size = 350
    colors = ['', 'cyan', 'orange', '']

    manipulate3d.draw_tetrahedron(window, center_x, center_y, size, colors)

    manipulate3d.infinite_rotation(window, center_x, center_y, colors)
