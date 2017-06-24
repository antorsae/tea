import numpy as np


class OccupancyGrid:
    class Cell:
        DefaultValue = 0

        def __init__(self):
            self.indices = []
            self.minZ = 1e6
            self.maxZ = -1e6

        def update(self, index, point):
            self.indices.append(index)

            z = point[2]
            self.minZ = min(self.minZ, z)
            self.maxZ = max(self.maxZ, z)

        def getOccupValue(self, meth):
            point_num = len(self.indices)
            if meth == 'zVar':
                if point_num < 2:
                    return self.DefaultValue
                return self.maxZ - self.minZ
            else:
                return point_num > 0

        def empty(self):
            return not self.indices

    def __init__(self, gridSize, cellSize, cellMethod='zVar', verbose=False):
        self.gridSize = gridSize
        self.cellSize = cellSize

        N = int(gridSize / cellSize)
        self.gridCols = N

        self.cellMethod = cellMethod
        self.verbose = verbose

        if self.verbose:
            print("Grid size: {}x{}".format(N, N))

        self.cells = []

        self.maxCellOccupValue = 0

    def rebuild(self, cloud):
        self.cells = []
        for i in xrange(self.gridCols * self.gridCols):
            self.cells.append(self.Cell())

        assert isinstance(cloud, np.ndarray)

        points_num = cloud.shape[0]

        for pi in xrange(points_num):
            pnt = cloud[pi, :]

            if not self.isIn(pnt):
                continue

            cellIdx = self.cvtEgo2Gridi(pnt)

            self.cells[cellIdx].update(pi, pnt)

        occups = [c.getOccupValue(self.cellMethod) for c in self.cells]
        self.maxCellOccupValue = max(occups)

    def isIn(self, point):
        hs = self.gridSize/2
        x, y = point[:2]
        return -hs <= x < hs and -hs <= y < hs

    def cvtEgo2Grid(self, point):
        x, y = point[:2]
        col = int(x/self.cellSize) + int(self.gridCols/2)
        row = int(self.gridCols/2) - int(y/self.cellSize)
        assert row < self.gridCols and col < self.gridCols
        return row, col

    def cvtEgo2Gridi(self, point):
        row, col = self.cvtEgo2Grid(point)
        return col + self.gridCols * row

    def rasterize(self, scale=1., logVal=True):
        data = np.zeros((self.gridCols, self.gridCols), np.float32)

        if self.maxCellOccupValue == OccupancyGrid.Cell.DefaultValue:
            print("warning: all occupancy values are zero (not enough points)")
            return data

        nCells = self.gridCols ** 2

        for ic in xrange(nCells):
            row, col = ic / self.gridCols, ic % self.gridCols
            val = self.cells[ic].getOccupValue(self.cellMethod)

            normVal = np.log(val + 1.) / np.log(self.maxCellOccupValue + 1.) if logVal\
                else float(val) / self.maxCellOccupValue

            data[row, col] = scale * normVal

        return data