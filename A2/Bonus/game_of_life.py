"""
conway.py

A simple Python/matplotlib implementation of Conway's Game of Life.

Author: Mahesh Venkitachalam
"""

import argparse
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ON = 1
OFF = 0
vals = [ON, OFF]


def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[0.2, 0.8]).reshape(N, N)


def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
    grid[i : i + 3, j : j + 3] = glider


def addGosperGliderGun(i, j, grid):
    """adds a Gosper Glider Gun with top left cell at (i, j)"""
    gun = np.zeros(11 * 38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 1
    gun[6][1] = gun[6][2] = 1

    gun[3][13] = gun[3][14] = 1
    gun[4][12] = gun[4][16] = 1
    gun[5][11] = gun[5][17] = 1
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 1
    gun[7][11] = gun[7][17] = 1
    gun[8][12] = gun[8][16] = 1
    gun[9][13] = gun[9][14] = 1

    gun[1][25] = 1
    gun[2][23] = gun[2][25] = 1
    gun[3][21] = gun[3][22] = 1
    gun[4][21] = gun[4][22] = 1
    gun[5][21] = gun[5][22] = 1
    gun[6][23] = gun[6][25] = 1
    gun[7][25] = 1

    gun[3][35] = gun[3][36] = 1
    gun[4][35] = gun[4][36] = 1

    grid[i : i + 11, j : j + 38] = gun


def update(frameNum, img, grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line
    new_grid = grid.copy()
    neighbors_count = (
        np.roll(grid, 1, axis=0)
        + np.roll(grid, -1, axis=0)
        + np.roll(grid, 1, axis=1)
        + np.roll(grid, -1, axis=1)
        + np.roll(grid, (1, 1), axis=(0, 1))
        + np.roll(grid, (-1, -1), axis=(0, 1))
        + np.roll(grid, (-1, 1), axis=(0, 1))
        + np.roll(grid, (1, -1), axis=(0, 1))
    ) // 1
    survivor = ((neighbors_count == 2) | (neighbors_count == 3)) & (grid == ON)
    new_born = (neighbors_count == 3) & (grid == OFF)
    new_grid = np.where(new_born | survivor, ON, OFF).astype(int)
    # update data
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return (img,)


def run_one_step(grid, N):
    new_grid = grid.copy()
    neighbors_count = (
        np.roll(grid, 1, axis=0)
        + np.roll(grid, -1, axis=0)
        + np.roll(grid, 1, axis=1)
        + np.roll(grid, -1, axis=1)
        + np.roll(grid, (1, 1), axis=(0, 1))
        + np.roll(grid, (-1, -1), axis=(0, 1))
        + np.roll(grid, (-1, 1), axis=(0, 1))
        + np.roll(grid, (1, -1), axis=(0, 1))
    ) // 1
    survivor = ((neighbors_count == 2) | (neighbors_count == 3)) & (grid == ON)
    new_born = (neighbors_count == 3) & (grid == OFF)
    new_grid = np.where(new_born | survivor, ON, OFF).astype(int)
    return new_grid


def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Runs Conway's Game of Life simulation."
    )
    # add arguments
    parser.add_argument("--grid-size", dest="N", required=False)
    parser.add_argument("--mov-file", dest="movfile", required=False)
    parser.add_argument("--interval", dest="interval", required=False)
    parser.add_argument("--glider", action="store_true", required=False)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        required=False,
        help="Disable GUI and animation for performance testing.",
    )
    parser.add_argument("--gosper", action="store_true", required=False)
    args = parser.parse_args()

    # set grid size
    N = 100
    if args.N and int(args.N) > 8:
        N = int(args.N)

    # set animation update interval
    updateInterval = 50
    if args.interval:
        updateInterval = int(args.interval)

    # declare grid
    grid = np.array([])
    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N * N).reshape(N, N)
        addGlider(1, 1, grid)
    elif args.gosper:
        grid = np.zeros(N * N).reshape(N, N)
        addGosperGliderGun(10, 10, grid)
    else:
        # populate grid with random on/off - more off than on
        grid = randomGrid(N)

    # If --no-gui is set, just run the update loop for X iterations
    if args.no_gui:
        iterations = 50  # example fixed number of iterations
        for _ in range(iterations):
            grid = run_one_step(grid, N)
        sys.exit(0)  # end here

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation="nearest")
    ani = animation.FuncAnimation(
        fig,
        update,
        fargs=(
            img,
            grid,
            N,
        ),
        frames=10,
        interval=updateInterval,
        save_count=50,
    )

    # # of frames?
    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=["-vcodec", "libx264"])

    plt.show()


# call main
if __name__ == "__main__":
    main()
