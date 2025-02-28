from graphics import *


def set_color_width_and_draw(shape, window, r, g, b):
    shape.setOutline(color_rgb(r, g, b))
    shape.setWidth(2)
    shape.draw(window)


def draw_rectangle(rectangle, window):
    set_color_width_and_draw(rectangle, window, 0, 255, 50)


def draw_ellipse(ellipse, window):
    set_color_width_and_draw(ellipse, window, 0, 50, 255)


def draw_triangle_via_line(line, window):
    set_color_width_and_draw(line, window, 255, 0, 50)


if __name__ == '__main__':
    # Task-1
    win = GraphWin("Task-1 Ellipses, Rectangles - graphics module built in Objects used", 1000, 1000)

    ellipse1 = Oval(Point(200, 400), Point(800, 0))
    ellipse2 = Oval(Point(225, 375), Point(775, 25))
    ellipse3 = Oval(Point(250, 350), Point(750, 50))
    ellipse4 = Oval(Point(275, 325), Point(725, 75))
    ellipse5 = Oval(Point(300, 300), Point(700, 100))

    rectangle1 = Rectangle(Point(100, 700), Point(900, 400))
    rectangle2 = Rectangle(Point(125, 675), Point(875, 425))
    rectangle3 = Rectangle(Point(150, 650), Point(850, 450))
    rectangle4 = Rectangle(Point(175, 625), Point(825, 475))
    rectangle5 = Rectangle(Point(200, 600), Point(800, 500))

    draw_ellipse(ellipse1, win)
    draw_ellipse(ellipse2, win)
    draw_ellipse(ellipse3, win)
    draw_ellipse(ellipse4, win)
    draw_ellipse(ellipse5, win)

    draw_rectangle(rectangle1, win)
    draw_rectangle(rectangle2, win)
    draw_rectangle(rectangle3, win)
    draw_rectangle(rectangle4, win)
    draw_rectangle(rectangle5, win)

    win.getMouse()  # Pause to view result
    win.close()  # Close window when done

    # Task-2.1
    # --------------------------------------------------------------------------------------
    win2 = GraphWin("Task-2 Triangles - Using graphics lines", 1000, 1000)

    line1 = Line(Point(350, 50), Point(650, 50))
    line2 = Line(Point(350, 50), Point(650, 950))
    line3 = Line(Point(350, 950), Point(650, 50))
    line4 = Line(Point(350, 950), Point(650, 950))

    line5 = Line(Point(50, 350), Point(50, 650))
    line6 = Line(Point(50, 350), Point(950, 650))
    line7 = Line(Point(50, 650), Point(950, 350))
    line8 = Line(Point(950, 350), Point(950, 650))

    draw_triangle_via_line(line1, win2)
    draw_triangle_via_line(line2, win2)
    draw_triangle_via_line(line3, win2)
    draw_triangle_via_line(line4, win2)

    draw_triangle_via_line(line5, win2)
    draw_triangle_via_line(line6, win2)
    draw_triangle_via_line(line7, win2)
    draw_triangle_via_line(line8, win2)

    win2.getMouse()
    win2.close()

    # Task-2.2
    # --------------------------------------------------------------------------------------
    win3 = GraphWin("Task-2.2 Triangles filled monochrome", 1000, 1000)

    triangle_outer = Polygon(Point(350, 50), Point(650, 50), Point(500, 500))
    triangle_outer.setOutline(color_rgb(255, 255, 0))
    triangle_outer.setWidth(2)
    triangle_outer.setFill(color_rgb(50, 50, 50))
    triangle_outer.draw(win3)

    triangle_inner = Polygon(Point(350, 50), Point(650, 50), Point(500, 350))
    triangle_inner.setOutline(color_rgb(150, 150, 150))
    triangle_inner.setWidth(2)
    triangle_inner.setFill(color_rgb(200, 20, 200))
    triangle_inner.draw(win3)

    win3.getMouse()
    win3.close()

    # Task-2.3
    # --------------------------------------------------------------------------------------
    win4 = GraphWin("Task-2.3 Hexagon - Color", 1000, 1000)

    hexagon = Polygon(Point(325, 200),
                      Point(625, 200),
                      Point(775, 450),
                      Point(625, 700),
                      Point(325, 700),
                      Point(175, 450))

    hexagon.setOutline(color_rgb(255, 0, 255))
    hexagon.setWidth(4)
    hexagon.setFill(color_rgb(0, 255, 255))
    hexagon.draw(win4)

    win4.getMouse()
    win4.close()
