from graphics import *
from itertools import cycle
import time

positions = []
player = cycle(["cross", "circle"])

def drawBoard(window):
    vertLine1 = Line(Point((window.getWidth() / 2) - 50, 50), Point((window.getWidth() / 2) - 50, window.getHeight() - 50))
    vertLine1.setWidth(3)
    vertLine1.draw(window)

    vertLine2 = Line(Point((window.getWidth() / 2) + 50, 50), Point((window.getWidth() / 2) + 50, window.getHeight() - 50))
    vertLine2.setWidth(3)
    vertLine2.draw(window)

    horizLine1 = Line(Point(50, (window.getHeight() / 2) + 50), Point(window.getWidth() - 50, (window.getHeight() / 2) + 50))
    horizLine1.setWidth(3)
    horizLine1.draw(window)

    horizLine2 = Line(Point(50, (window.getHeight() / 2) - 50), Point(window.getWidth() - 50, (window.getHeight() / 2) - 50))
    horizLine2.setWidth(3)
    horizLine2.draw(window)

def drawSymbol(window, x , y):

    if (50 <= x < 150) and (50 <= y < 150) and 1 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(60, 60), Point(145,145))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(60, 145), Point(145, 60))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(100, 100), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(1)
    elif (150 <= x < 250) and (50 <= y < 150) and 2 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(155,60), Point(245,145))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(155, 145), Point(245, 60))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(200, 100), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(2)
    elif (250 <= x < 350) and (50 <= y < 150) and 3 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(255, 60), Point(345,145))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(255, 145), Point(345, 60))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(300, 100), 45)
            circle.setWidth(2)
            circle.draw(window)
        positions.append(3)
    elif (50 <= x < 150) and (150 <= y < 250) and 4 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(60, 160), Point(145,245))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(60, 245), Point(145, 160))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(100, 200), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(4)
    elif (150 <= x < 250) and (150 <= y < 250) and 5 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(155,160), Point(245,245))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(155, 245), Point(245, 160))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(200, 200), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(5)
    elif (250 <= x < 350) and (150 <= y < 250) and 6 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(255, 160), Point(345,245))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(255, 245), Point(345, 160))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(300, 200), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(6)
    elif (50 <= x < 150) and (250 <= y < 350) and 7 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(60, 260), Point(145, 345))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(60, 345), Point(145, 260))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(100, 300), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(7)
    elif (150 <= x < 250) and (250 <= y < 350) and 8 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(155, 260), Point(245, 345))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(155, 345), Point(245, 260))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(200, 300), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(8)
    elif (250 <= x < 350) and (250 <= y < 350) and 9 not in positions:
        currentPlayer = player.__next__()
        if currentPlayer == "cross":
            line1 = Line(Point(255, 260), Point(345, 345))
            line1.setWidth(2)
            line1.setFill("blue")
            line1.draw(window)
            line2 = Line(Point(255, 345), Point(345, 260))
            line2.setWidth(2)
            line2.setFill("blue")
            line2.draw(window)
        elif currentPlayer == "circle":
            circle = Circle(Point(300, 300), 45)
            circle.setWidth(2)
            circle.setOutline("red")
            circle.draw(window)
        positions.append(9)
    else:
        print("nothing to do")


def eventLoop(window):
    notquit = True

    while (window.checkKey() != 'q'):
        mouse = window.checkMouse()
        if(mouse != None):
            drawSymbol(window, mouse.getX(), mouse.getY())

        # 60 FPS
        time.sleep(.016)

    window.close()

def main():

    window = GraphWin('Morpion', 400, 400) # give title and dimensions

    # Draw Board
    drawBoard(window)


    # Draw Instruction
    message = Text(Point(window.getWidth()/2, window.getHeight()-20), 'Click to play.')
    message.draw(window)

    # Process events
    eventLoop(window)

main()