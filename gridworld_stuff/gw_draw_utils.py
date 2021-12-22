# Author: Aaron Jacobson
# Created: 11/7/2021
# Email: AaronMorauski@gmail.com

import math
import pygame as pg


class GridWorldImg:
    """Used for graphically displaying grid-worlds. Begins by drawing an undecorated grid, then allows for coloration
        of grid adjacent_tiles, addition of lines between adjacent_tiles, and the construction of a path.

        Lines are not required to be contiguous, and as many lines as fit on the grid may be created.

        The path structure is created as a list of adjacent_tiles' coordinates, and between the corresponding adjacent_tiles is drawn
        a path. Each GridWorldImg object can store only a single path at once, though multiple may be drawn on screen
        if the path points are updated between draws.

        Note that <default_circle_ratio> describes default circle diameter as a proportion of <tile_size>.

        All locations are tuples of the form (x, y), where x and y are the indices of the desired tile.
        The lowermost, leftmost tile has the index (0, 0).

        Known limitation: use of pygame restricts this module to displaying one grid-world at a time. Multiple
            Gridworld objects may coexist, but switching which is displayed may cause problems if the two grid-worlds
            call for differently sized windows. This may be partially fixable with further development."""

    def __init__(self, width_units: int = 10, height_units: int = 10, tile_size: int = 95,
                 margin: int = 5, background: tuple = (50, 50, 50), inactive_tile_color: tuple = (100, 100, 100),
                 active_tile_color: tuple = (70, 70, 200), default_line_color: tuple = (200, 200, 200),
                 default_line_width: int = 3, path_color: tuple = None, default_path_width: int = 3,
                 default_circle_color: tuple = (200, 200, 200), default_circle_ratio: float = .65,
                 title: str = "Grid World", font_size: int = None, default_text_color: tuple = (255, 255, 255)):

        # Screen setup
        self.width_units = width_units
        self.height_units = height_units
        self.screen_width = width_units*(margin+tile_size) + margin
        self.screen_height = height_units*(margin+tile_size) + margin
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.title = title
        pg.display.set_caption(title)

        # Colors
        self.background = background
        self.inactive_tile_color = inactive_tile_color
        self.active_tile_color = active_tile_color
        self.default_line_color = default_line_color
        if path_color is None:
            self.path_color = default_line_color
        else:
            self.path_color = path_color
        self.default_circle_color = default_circle_color
        self.default_text_color = default_text_color

        # Active element containers
        self.tiles = {}
        self.lines = {}
        self.path = [].copy()
        self.circles = {}
        self.annotations = {}

        # Element sizes
        self.tile_size = tile_size
        self.path_width = default_path_width
        self.line_width = default_line_width
        self.default_circle_ratio = default_circle_ratio
        self.default_circle_radius = (default_circle_ratio * self.tile_size) / 2
        self.margin = margin

        # Font initialization
        pg.font.init()
        if font_size is None:
            self.font_size = int(self.tile_size/5)
        else:
            self.font_size = font_size
        self.font = pg.font.SysFont('arial', self.font_size)

        # Create screen with blank grid-world
        self.update_screen()

    def __repr__(self):
        to_return = 'Rows: {}; columns: {}\nadjacent_tiles: {};\nLines: {};\nPath: {};\nCircles:{}'.format(self.height_units,
                                                                                                  self.width_units,
                                                                                                  self.tiles,
                                                                                                  self.lines,
                                                                                                  self.path,
                                                                                                  self.circles)
        return to_return

    @staticmethod
    def main():
        """This is a loop that keeps the grid-world window responsive. Put this at the end of code that modifies
            the grid-world to keep the window from crashing."""
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    pg.display.quit()

    @staticmethod
    def sustain():
        """This is a helper method inserted here and there to keep the pygame window from becoming unresponsive."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.display.quit()

    # MAIN METHODS #
    def reset_all(self):
        """Clear all active adjacent_tiles and lines, delete the current path, and re-draw a fresh grid."""
        self.tiles_clear()
        self.lines_clear()
        self.path_clear()
        self.lines_clear()
        self.annotations_clear()
        self.screen.fill(self.background)
        for y in range(self.height_units):
            for x in range(self.width_units):
                self.tile_color((x, y), active=False, suppress=True)
        pg.display.set_caption(self.title)
        pg.display.update()

    def update_screen(self):
        """Draw the current state of the grid-world. This resets the display and re-draws all active adjacent_tiles,
            all active lines, and the active path. This may be optimized in the future.
            Draw order is: adjacent_tiles, circles, lines, path, annotations. Keep this in mind if layering elements."""
        # Resetting grid
        self.screen.fill(self.background)
        for y in range(self.height_units):
            for x in range(self.width_units):
                self.tile_color((x, y), active=False, suppress=True)
        # Re-drawing active elements
        for location in self.tiles:
            self.tile_color(location, self.tiles[location])
        for location in self.circles:
            self.circle_draw(location, self.circles[location][0], self.circles[location][1])
        for coordinates in self.lines:
            self.line_draw(coordinates[0], coordinates[1], self.lines[coordinates])
        if len(self.path) > 1:
            self.path_draw(self.path_color, self.path_width)
        for location in self.annotations:
            self.annotate(location, self.annotations[0], self.annotations[1])
        pg.display.update()
        pg.display.set_caption(self.title)
        pg.image.save(self.screen,'plots/Gridworld/gridworld.png')

    def get_corner_coords(self, location: tuple):
        """Helper method for drawing methods [ i.e. self.tile_color() ].
            Gets the screen coordinates of the corner of the tile at indices (x, y).

            Note that indices may be given as floats to get coordinates between corners."""
        x = self.margin + (self.margin + self.tile_size) * location[0]
        y = self.margin + (self.margin+self.tile_size) * (self.height_units - location[1] - 1)
        return x, y

    def get_center_coords(self, location: tuple):
        """Helper method for drawing methods [ i.e. self.line_draw() and self.path_draw() ].
            Gets the screen coordinates of the center of the tile at indices (x, y).

            Note that indices may be given as floats to get coordinates between centers."""
        x = self.margin + (self.margin+self.tile_size)*(location[0]) + .5*self.tile_size
        y = self.margin + (self.margin+self.tile_size)*(self.height_units - location[1] - 1) + .5*self.tile_size
        return x, y

    # TILES #
    def tile_add(self, location: tuple, color: tuple = None):
        """Add an active tile. If no color is given, <active_tile_color> will be used.
            Adding a tile where one already exists will overwrite the old color value.

            The added tile will not appear until the screen is updated."""
        if color is None:
            color = self.active_tile_color
        self.tiles[location] = color

    def tile_delete(self, location: tuple):
        """Delete an active tile at <location>; it will appear active until the screen is updated."""
        if location in self.tiles:
            self.tiles.pop(location)

    def tiles_clear(self):
        """Delete all active adjacent_tiles; they will appear active until the screen is updated."""
        self.tiles = {}

    def tile_color(self, location: tuple, color: tuple = None, active: bool = True, suppress: bool = False):
        """Apply color to a tile. If no color is given, one of two defaults will be used:
            If <active> is True, then <active_tile_color> will be used. Else, <inactive_tile_color> will be used.

            Especially useful for updating the display without re-drawing the entire grid.

            Does NOT add selected tile to active adjacent_tiles."""
        if color is None:
            if active:
                color = self.active_tile_color
            else:
                color = self.inactive_tile_color
        coordinates = self.get_corner_coords(location)
        new_rect = pg.draw.rect(self.screen,
                                pg.Color(color),
                                pg.Rect(coordinates[0], coordinates[1], self.tile_size, self.tile_size))
        self.sustain()
        if not suppress:
            pg.display.update(new_rect)

    def tile_reset(self, location: tuple):
        """Remove an active tile; it will immediately appear inactive.

            Best when used without lines or paths; this allows adjacent_tiles to be reset without re-drawing the entire grid.

            Does NOT remove selected tile from active adjacent_tiles."""
        if location in self.tiles:
            self.tile_color(location, active=False)
            self.tiles.pop(location)

    # LINES #
    def line_add(self, start: tuple, stop: tuple, color: tuple = None):
        """Add an active line from the tile at <start> to the tile at <stop> (center of tile is used as endpoint).
            <start> and <stop> are tuples of the form (x, y), where x and y are the indices of a tile.

            If no color is given, <default_line_color> will be used.

            The added line will not appear until the screen is updated."""
        if color is None:
            color = self.default_line_color
        self.lines[(start, stop)] = color

    def line_remove(self, start: tuple, stop: tuple):
        """Delete an active line; it will appear active until the screen is updated."""
        self.lines.pop((start, stop))

    def lines_clear(self):
        """Delete all active lines; they will appear active until the screen is updated."""
        self.lines = {}

    def line_draw(self, start: tuple, stop: tuple, color: tuple = None, width: int = None):
        """Draw a line on the grid from the tile at <start> to the tile at <stop>.
            <start> and <stop> are tuples of the form (x, y), where x and y are the indices of a tile.

            If no color is given, <default_line_color> will be used."""
        if color is None:
            color = self.default_line_color
        if width is None:
            width = self.line_width
        start_coordinates = self.get_center_coords(start)
        stop_coordinates = self.get_center_coords(stop)
        new_line = pg.draw.line(self.screen, color, start_coordinates, stop_coordinates, width=width)
        self.sustain()
        pg.display.update(new_line)

    # PATH #
    def path_extend(self, location: tuple):
        """Add a node to the current path. Accepts a tuple (x, y) where x and y are the indices of a tile in the grid.
            The added node will not appear until the screen is updated."""
        coordinates = self.get_center_coords(location)
        self.path.append(coordinates)

    def path_backtrack(self, n: int = 1):
        """Remove the most recent <n> nodes from the current path. By default, <n> is 1.
            These nodes will remain visible until the screen is reset."""
        for step in range(n):
            self.path.pop()

    def path_clear(self):
        """Delete all nodes from the current path. The current path will remain visible until the screen is updated."""
        self.path = [].copy()

    def path_draw(self, color: tuple = None, width: int = None):
        """Draw the the current path on the grid. If no color is given, <path_color> will be used."""
        if color is None:
            color = self.path_color
        if width is None:
            width = self.path_width
        new_path = pg.draw.lines(self.screen, color, False, self.path, width=width)
        self.sustain()
        pg.display.update(new_path)

    # CIRCLES #
    def circle_add(self, location: tuple, color: tuple = None, radius_ratio: float = None):
        """Add an active circle. If no color is given, <default_circle_color> will be used.
            Adding a circle where one already exists will overwrite the old circle.

            The added circle will not appear until the screen is updated."""
        if color is None:
            color = self.default_circle_color
        if radius_ratio is None:
            radius_ratio = self.default_circle_ratio
        self.circles[location] = (color, radius_ratio)

    def circle_delete(self, location: tuple):
        """Delete an active circle at <location>; it will appear active until the screen is updated."""
        if location in self.tiles:
            self.circles.pop(location)

    def circles_clear(self):
        """Delete all active circles; they will appear active until the screen is updated."""
        self.circles = {}

    def circle_draw(self, location: tuple, color: tuple = None, radius_ratio: float = None):
        """Draw a circle at <location>. If no color is given, <default_circle_color> will be used.
            <radius_ratio> determines circle diameter as a proportion of <tile_size>."""
        if color is None:
            color = self.default_circle_color
        if radius_ratio is None:
            radius = self.default_circle_radius
        else:
            radius = (radius_ratio * self.tile_size) / 2
        coordinates = self.get_center_coords(location)
        new_circle = pg.draw.circle(self.screen, color, coordinates, radius)
        self.sustain()
        pg.display.update(new_circle)

    def annotation_add(self, location: tuple, text: str, color: tuple = None):
        """Add an active annotation. If no color is given, <default_text_color> will be used.
            Adding an annotation where one already exists will overwrite the old annotation.

            The added annotation will not appear until the screen is updated."""
        if color is None:
            color = self.default_text_color
        self.annotations[location] = (color, text)

    def annotation_delete(self, location: tuple):
        """Delete an active annotation at <location>; it will appear active until the screen is updated."""
        if location in self.annotations:
            self.annotations.pop(location)

    def annotations_clear(self):
        """Delete all active annotations; they will appear active until the screen is updated."""
        self.annotations = {}

    def annotate(self, location: tuple, text: str, color: tuple = None):
        """Used for annotating the grid-world. Location is given by tile coordinates, but need not be integers.
            By default, this uses the <font_size> associated with the GridWorldImg object.
            Font size defaults to <tile_size>/5. <color> defaults to white."""
        if color is None:
            color = self.default_text_color
        location = (self.get_corner_coords(location)[0] + 1, self.get_corner_coords(location)[1] - 1)
        new_annotation = self.screen.blit(self.font.render(text, False, color), location)
        self.sustain()
        pg.display.update(new_annotation)


if __name__ == '__main__':
    print('kwa kwa')
    # A few examples here

    # gw1 = GridWorldImg()
    # for i in list(range(4)):
    #     for j in range(3):
    #         gw1.tile_add((i, j))
    # points = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)]
    # for point in points:
    #     gw1.path_extend(point)
    # gw1.circle_add((4, 5))
    # gw1.circle_add((7, 7), (0, 150, 100), .3)
    # gw1.update_screen()
    # gw1.annotate((5, 5), 'Text')
    # gw1.annotate((5, 4.75), 'More Text')
    # gw1.main()

    # gw2 = GridWorldImg(18, 14, 45, 1)
    # gw2.tile_color((3, 5), (200, 0, 0))
    # gw2.circle_draw((7, 9), (0, 255, 100), .5)
    # gw2.circle_draw((5.5, 5.5), radius_ratio=.25)
    # gw2.tile_color((13, 9), (100, 0, 0))
    # gw2.line_draw((17, 13), (17, 9))
    # gw2.line_draw((17, 9), (14, 3))
    # gw2.line_draw((14, 2), (14.75, 2.75))
    # gw2.path_extend((4, 2))
    # gw2.path_extend((7, 2))
    # gw2.path_extend((8, 4))
    # gw2.path_draw(color=(200, 140, 170), width=7)
    # gw2.main()

    gw3 = GridWorldImg(75, 75, 8, 1)
    for i in range(30):
        gw3.tile_add((i+5, i*2 + 15), (100, 255, 0))
        gw3.circle_add((75-i, i+20), (200, 80, 255))
    for i in range(50):
        gw3.tile_add((10+i, 71-math.floor(math.sqrt(5*i))))
    for i in range(100):
        gw3.path_extend((i*math.cos(i/8)/3+35, i*math.sin(i/8)/3+35))
    for i in reversed(range(100)):
        gw3.path_extend((i*math.cos(i/16)/3+35, i*math.sin(i/16)/3+35))
    gw3.update_screen()
    gw3.main()
