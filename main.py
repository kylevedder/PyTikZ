from matplotlib.pyplot import gray, text
import tikz
from math import cos, sin, radians, acos, asin
from enum import Enum
import random
import numpy as np


class Direction(Enum):
    RIGHT = 1
    LEFT = -1


class Parallelogram(Enum):
    VERT = 1
    HORZ = -1


pic = tikz.Picture('thick, scale=0.9')

# with pic.path('draw') as draw:
#     draw.at(0 + 0j).grid_to(3 + 3j)


def color_tuple_to_str(color):
    assert len(color) == 3
    return f"{{rgb,255:red,{color[0]};green,{color[1]};blue,{color[2]}}}"


def draw_arrow(start_pt, end_pt):
    with pic.path('->, very thick, draw') as draw:
        draw.at_pt(start_pt).line_to_pt(end_pt)


def draw_multi_point_arrow(points):
    start_pt = points[0]
    with pic.path('->, very thick, draw') as draw:
        state = draw.at_pt(start_pt)
        for pt in points[1:]:
            state.line_to_pt(pt)


def draw_text(center_pt,
              string,
              font_size=None,
              line_height=0.4,
              color="black"):
    assert type(center_pt) == tuple
    assert len(center_pt) == 2
    assert type(string) == str

    font_size_string = ""
    style_string = ""
    if font_size is not None:
        font_size_string = f"\\fontsize{{{font_size}pt}}{{{font_size}pt}}"
        style_string = f"font={font_size_string}"

    lines = string.split("\n")
    print(lines)
    max_y_offset = (len(lines) - 1) / 2 * line_height
    print("max_y_offset", max_y_offset)

    x, y = center_pt
    with pic.path(f'draw, color={color}') as draw:
        for idx, l in enumerate(lines):
            line_center_offset = max_y_offset - idx * line_height
            line_center = x, y + line_center_offset
            print(line_center)
            draw.at_pt(line_center).node(
                f"{{{font_size_string}$\\textup{{{l}}}$}}", style=style_string)


def get_points_center(points):
    """
    Compute the average point from the list of point tuples.
    """
    x = sum([x for x, _ in points]) / len(points)
    y = sum([y for _, y in points]) / len(points)
    return x, y


def draw_parallelogram(
    minx: float,
    miny: float,
    tile_width: float,
    tile_height: float,
    slide: float,
    p_type: Parallelogram,
    direction: Direction = Direction.LEFT,
    global_x: float = 0,
    global_y: float = 0,
    color="black",
    fill_color="white",
    text="",
    font_size=None,
    pattern_dir=None,
    pattern_color=None,
    line_height=0.4,
):
    assert direction in [Direction.LEFT, Direction.RIGHT]

    if type(color) == tuple:
        color = color_tuple_to_str(color)
    if p_type == Parallelogram.HORZ:
        corner1 = minx + global_x, miny + global_y
        corner2 = minx + global_x + tile_width, miny + global_y
        corner3 = minx + global_x + tile_width + slide, miny + global_y + tile_height
        corner4 = minx + global_x + slide, miny + global_y + tile_height
    elif p_type == Parallelogram.VERT:
        corner1 = minx + global_x, miny + global_y
        corner2 = minx + global_x, miny + global_y + tile_height
        corner3 = minx + global_x + tile_width, miny + global_y + tile_height + slide
        corner4 = minx + global_x + tile_width, miny + global_y + slide
    else:
        raise ValueError(f"Parallelogram type {p_type} not supported")
    center_pt = get_points_center([corner1, corner2, corner3, corner4])
    if type(fill_color) == np.str_:
        fill_color = str(fill_color)
    assert type(
        fill_color
    ) == str, f"fill_color type {type(fill_color)} ({fill_color}) must be a string"
    style_str = f'draw, color={color}, fill={fill_color}'
    if pattern_dir is not None:
        style_str += f", pattern={pattern_dir}"
    if pattern_color is not None:
        style_str += f", pattern color={pattern_color}"
    with pic.path(style_str) as draw:
        draw.at_pt(corner1).line_to_pt(corner2).line_to_pt(corner3).line_to_pt(
            corner4).cycle()
    draw_text(center_pt,
              text,
              font_size=font_size,
              color=color,
              line_height=line_height)


def draw_rectangle(
    x,
    y,
    width,
    height,
    color="black",
    fill_color="white",
    text="",
    font_size=None,
    pattern_dir=None,
    pattern_color=None,
):
    draw_parallelogram(x,
                       y,
                       width,
                       height,
                       0,
                       Parallelogram.HORZ,
                       Direction.RIGHT,
                       color=color,
                       fill_color=fill_color,
                       text=text,
                       font_size=font_size,
                       pattern_dir=pattern_dir,
                       pattern_color=pattern_color)


def draw_angle_grid(grid_shape,
                    tile_width,
                    tile_height,
                    slide,
                    p_type: Parallelogram,
                    direction: Direction = Direction.LEFT,
                    color="black",
                    fill_color="white",
                    global_x=0,
                    global_y=0,
                    cell_text=None,
                    font_size=None):
    tiles_x, tiles_y = grid_shape
    for xidx in range(tiles_x):
        for yidx in range(tiles_y):
            if cell_text is not None:
                text = cell_text[xidx, yidx]
            else:
                text = ""
            # pos_x = xidx * yoffset
            # pos_y = yidx * tile_height
            if p_type == Parallelogram.HORZ:
                pos_x = xidx * tile_width + slide * yidx
                pos_y = yidx * tile_height
            elif p_type == Parallelogram.VERT:
                pos_x = xidx * tile_width
                pos_y = yidx * tile_height + slide * xidx
            else:
                raise ValueError(f"Parallelogram type {p_type} not supported")

            color_val = color
            fill_color_val = fill_color
            if type(color) == np.array or type(color) == np.ndarray:
                assert color.shape == grid_shape, f"color shape {color.shape} must be {grid_shape}"
                color_val = color[yidx, xidx]
            if type(fill_color) == np.array or type(fill_color) == np.ndarray:
                assert fill_color.shape == grid_shape, f"fill_color shape {fill_color.shape} must be {grid_shape}"
                fill_color_val = fill_color[xidx, yidx]

            draw_parallelogram(pos_x, pos_y, tile_width, tile_height, slide,
                               p_type, direction, global_x, global_y,
                               color_val, fill_color_val, text, font_size)


def draw_cube(tiles_x: int,
              tiles_y: int,
              tile_width: float,
              tile_height: float,
              cube_height: float,
              slide: float,
              color="black",
              fill_color="white",
              global_x: float = 0,
              global_y: float = 0,
              front_cell_text=None,
              font_size=None):
    top_fill_color = fill_color.copy(
    ) if type(fill_color) != str else fill_color
    front_fill_color = fill_color.copy(
    ) if type(fill_color) != str else fill_color
    right_fill_color = fill_color.copy(
    ) if type(fill_color) != str else fill_color
    if type(front_fill_color) == np.array or type(
            front_fill_color) == np.ndarray:
        front_fill_color = front_fill_color[:, 0].reshape(1, -1).T
    if type(right_fill_color) == np.array or type(
            right_fill_color) == np.ndarray:
        right_fill_color = np.array(right_fill_color[-1]).reshape(-1, 1)
    draw_angle_grid((tiles_x, tiles_y), tile_width, tile_height, slide,
                    Parallelogram.HORZ, Direction.LEFT, color, top_fill_color,
                    global_x, global_y - cube_height)
    draw_angle_grid((tiles_x, 1), tile_width, cube_height, 0.0,
                    Parallelogram.HORZ, Direction.LEFT, color,
                    front_fill_color, global_x, global_y - cube_height,
                    front_cell_text, font_size)
    draw_angle_grid((tiles_x, 1), slide, cube_height, tile_height,
                    Parallelogram.VERT, Direction.LEFT, color,
                    right_fill_color, global_x + tile_width * tiles_x,
                    global_y - cube_height)


def generate_points_random(floor_width, floor_height, slide):
    rng = random.Random(4)

    def gen_circle():
        x = rng.uniform(0, floor_width + slide)
        y = floor_height / (floor_width + slide) * x + rng.uniform(-0.2,
                                                                   0.8) + 0.2
        z = floor_height / (floor_width + slide) * x + rng.uniform(-0.2,
                                                                   0.8) + 0.2
        return x, y, z

    circles = [gen_circle() for _ in range(10)]
    circles.sort(key=lambda x: x[2])


def generate_points(floor_width):
    circles = [(0.2625955114992747, 0.20978201000275148, 0.31799915015013785),
               (0.4660491689496671, 0.6698369135890041, 0.4098395488556408),
               (0.2254417686850616, 0.15949845816103608, 0.494574376966521),
               (0.4957009884486125, 0.2447948880731765, 0.5376870964531417),
               (1.410573058684594, 0.9089746496202993, 0.5808110481506684),
               (1.680940460940229, 0.6737043211450198, 0.7901186604253152),
               (1.8481065778585188, 0.6147487034516415, 1.1338823345365994),
               (1.3166487650675733, 1.108080070326596, 1.2308337194392251),
               (1.9277055904842098, 1.3512253773484397, 1.3159356283580697),
               (1.947698825998927, 1.3854054276352556, 1.3631377255591597)]

    circles = [(x * (floor_width / 2), y, z) for x, y, z in circles]
    return circles


def point_to_grayscale(pt):
    x, y, z = pt
    grayscale = 255 - z * 100
    return grayscale


def draw_pointcloud(global_x, global_y, num_tiles, floor_width, floor_height,
                    slide, circle_size):
    # Draw floor
    draw_parallelogram(0,
                       0,
                       floor_width,
                       floor_height,
                       slide * num_tiles,
                       Parallelogram.HORZ,
                       Direction.LEFT,
                       color="black",
                       global_x=global_x,
                       global_y=global_y)

    for x, y, z in generate_points(floor_width):
        grayscale = point_to_grayscale((x, y, z))
        with pic.path(
                f'draw, fill={color_tuple_to_str((grayscale, grayscale, grayscale))}'
        ) as draw:
            draw.at_pt((x + global_x, y + global_y)).circle(circle_size)
    draw_text((global_x + (floor_width + slide) / 2,
               global_y - label_text_size_line_height * 1.5),
              "Raw\nPointcloud",
              font_size=label_text_size,
              line_height=label_text_size_line_height)


def draw_pillarized_pointcloud(global_x, global_y, num_tiles, floor_width,
                               floor_height, slide, cube_height, circle_size):

    pts = generate_points(floor_width)
    minx, maxx = min(pts, key=lambda x: x[0])[0], max(pts,
                                                      key=lambda x: x[0])[0]
    minz, maxz = min(pts, key=lambda x: x[2])[2], max(pts,
                                                      key=lambda x: x[2])[2]

    def point_to_col_idx(pt):
        """
        Use the X and Z coordinates of the point to decide what row and column index to draw it in.
        """
        x, y, z = pt
        x_idx = (x - minx) / (maxx - minx)
        z_idx = (z - minz) / (maxz - minz)
        return min(int(x_idx * num_tiles),
                   num_tiles - 1), min(int(z_idx * num_tiles), num_tiles - 1)

    fill_list = np.array([zero_fill_color for _ in range(num_tiles**2)],
                         dtype='object').reshape((num_tiles, num_tiles))
    for x, y, z in pts:
        x_idx, z_idx = point_to_col_idx((x, y, z))
        assert x_idx < num_tiles, f"x_idx={x_idx} num_tiles={num_tiles}"
        assert z_idx < num_tiles, f"z_idx={z_idx} num_tiles={num_tiles}"
        fill_list[z_idx, x_idx] = non_zero_fill_color

    draw_cube(num_tiles,
              num_tiles,
              floor_width / num_tiles,
              floor_height / num_tiles,
              -cube_height,
              slide,
              color="black",
              fill_color=fill_list,
              global_x=global_x,
              global_y=global_y)

    for x, y, z in pts:
        grayscale = point_to_grayscale((x, y, z))
        x_idx, z_idx = point_to_col_idx((x, y, z))
        if z_idx != 0:
            continue
        with pic.path(
                f'draw, fill={color_tuple_to_str((grayscale, grayscale, grayscale))}'
        ) as draw:
            draw.at_pt((x + global_x, y + global_y)).circle(circle_size)

    draw_text((global_x + (floor_width + slide) / 2,
               global_y - label_text_size_line_height * 1.5),
              "Pillarized\nPointcloud",
              font_size=label_text_size,
              line_height=label_text_size_line_height)
    return fill_list


def draw_gather_vectors(global_x,
                        global_y,
                        num_tiles,
                        matrix_width,
                        matrix_height,
                        coord_width,
                        font_size=None):
    font_size_string = ""
    if font_size is not None:
        font_size_string = f"\\fontsize{{{font_size}pt}}{{{font_size}pt}}"

    y_slide_up = matrix_height * 1
    draw_angle_grid((1, num_tiles),
                    matrix_width,
                    matrix_height,
                    0.0,
                    Parallelogram.HORZ,
                    Direction.LEFT,
                    color="black",
                    fill_color=non_zero_fill_color,
                    global_x=global_x + coord_width,
                    global_y=global_y + y_slide_up)
    idxs = [(0, 0), (0, 1), (3, 1), (4, 2)]

    for yidx in range(num_tiles):
        ypos = (
            num_tiles - 1
        ) * matrix_height + matrix_height / 2 + global_y + y_slide_up - yidx * matrix_height
        xpos = global_x
        coo_x, coo_y = idxs[yidx]
        draw_text((xpos, ypos), f"({coo_x}, {coo_y})", font_size=font_size)

    for x, y, z in reversed([
        (1.35, 0.01, 0.5),
        (1.3, 0.04, 0.3),
        (matrix_width / 2, -matrix_width / num_tiles * 1.5 + 0.04, 0.3),
        (0.6, -matrix_width / num_tiles * 2.5 + 0.09, 1.2),
        (0.4, -matrix_width / num_tiles * 3.5 + 0.12, 1.4),
        (0.45, -matrix_width / num_tiles * 3.5 + 0.09, 1.4),
    ]):
        point_y_offset = (
            num_tiles -
            1) * matrix_height + matrix_height / 2 + global_y + y_slide_up
        point_x_offset = global_x + coord_width
        grayscale = point_to_grayscale((x, y, z))
        with pic.path(
                f'draw, fill={color_tuple_to_str((grayscale, grayscale, grayscale))}'
        ) as draw:
            draw.at_pt(
                (x + point_x_offset, y + point_y_offset)).circle(circle_size)

    draw_text((global_x + matrix_width / 2 + coord_width,
               global_y + y_slide_up - matrix_height * 0.5 + 0.05),
              "\\scriptsize\\vdots")

    draw_text((global_x + matrix_width / 2 + coord_width,
               global_y - label_text_size_line_height * 2),
              "\emph{Gather}ed\nNon-Empty Pillars\nin COO format",
              font_size=label_text_size,
              line_height=label_text_size_line_height)


def draw_vectorized_vectors(global_x,
                            global_y,
                            num_tiles,
                            matrix_width,
                            matrix_height,
                            coord_width,
                            font_size=None):
    font_size_string = ""
    if font_size is not None:
        font_size_string = f"\\fontsize{{{font_size}pt}}{{{font_size}pt}}"
    y_slide_up = matrix_height * 1
    draw_angle_grid((1, num_tiles),
                    matrix_width,
                    matrix_height,
                    0.0,
                    Parallelogram.HORZ,
                    Direction.LEFT,
                    color="black",
                    fill_color=non_zero_fill_color,
                    global_x=global_x + coord_width,
                    global_y=global_y + y_slide_up)
    idxs = [(0, 0), (0, 1), (3, 1), (4, 2)]

    for yidx in range(num_tiles):
        ypos = (
            num_tiles - 1
        ) * matrix_height + matrix_height / 2 + global_y + y_slide_up - yidx * matrix_height
        xpos = global_x
        coo_x, coo_y = idxs[yidx]
        draw_text((xpos, ypos), f"({coo_x}, {coo_y})", font_size=font_size)
        xpos = global_x + matrix_width / 2 + coord_width
        with pic.path(f'draw') as draw:
            draw.at_pt((xpos, ypos)).node(
                f"{{{font_size_string}$\\mathbb{{R}}^{{ \\scalebox{{0.5}}{{$ \\scriptscriptstyle N$}}}}$}}"
            )

    # with pic.path(f'draw') as draw:
    #     draw.at_pt(
    #         (global_x + matrix_width / 2 + coord_width, global_y + y_slide_up -
    #          matrix_height * 0.5)).node(f"{{{font_size_string}$\\vdots$}}")
    draw_text((global_x + matrix_width / 2 + coord_width,
               global_y + y_slide_up - matrix_height * 0.5 + 0.05),
              "\\scriptsize\\vdots")
    draw_text((global_x + matrix_width / 2 + coord_width,
               global_y - label_text_size_line_height * 2),
              "Vectorized\nPillars\nin COO format",
              font_size=label_text_size,
              line_height=label_text_size_line_height)


def draw_dense_pseudoimage(global_x,
                           global_y,
                           num_tiles,
                           floor_width,
                           fill_list,
                           font_size=None):
    text_list = np.array(
        [f"$\\vec{{\\mathbf{{0}}}}$" for _ in range(num_tiles**2)],
        dtype="object").reshape(fill_list.shape)
    text_list[
        fill_list !=
        zero_fill_color] = f"$\\mathbb{{R}}^{{\\scalebox{{0.5}}{{$ \\scriptscriptstyle N$}}}}$"
    print(text_list)
    draw_angle_grid((num_tiles, num_tiles),
                    floor_width / num_tiles,
                    floor_width / num_tiles,
                    0.0,
                    Parallelogram.HORZ,
                    Direction.LEFT,
                    color="black",
                    cell_text=text_list,
                    fill_color=fill_list,
                    global_x=global_x,
                    global_y=global_y,
                    font_size=font_size)
    draw_text((global_x + floor_width / 2, global_y + floor_width + 0.4),
              "\emph{Scatter}ed\n into Dense\nPseudoimage",
              font_size=label_text_size,
              line_height=label_text_size_line_height)


def draw_orig_backbone(global_x, global_y, backbone_width, backbone_height):
    draw_parallelogram(global_x,
                       global_y,
                       backbone_width,
                       backbone_height,
                       0.0,
                       Parallelogram.HORZ,
                       Direction.LEFT,
                       text="Original Dense\nBackbone",
                       font_size=label_text_size,
                       line_height=label_text_size_line_height)


def draw_sparse_backbone(global_x, global_y, backbone_width, backbone_height):
    draw_parallelogram(global_x,
                       global_y,
                       backbone_width,
                       backbone_height,
                       0.0,
                       Parallelogram.HORZ,
                       Direction.LEFT,
                       text="Our Sparse\nBackbone",
                       font_size=label_text_size,
                       line_height=label_text_size_line_height)


def draw_ssd(global_x, global_y, ssd_width, ssd_height):
    draw_parallelogram(global_x,
                       global_y,
                       ssd_width,
                       ssd_height,
                       0.0,
                       Parallelogram.HORZ,
                       Direction.LEFT,
                       text="Single\nStage\nDetector",
                       font_size=label_text_size,
                       line_height=label_text_size_line_height)


def draw_bounding_boxes(global_x, global_y, floor_width, floor_height,
                        num_tiles, slide):
    draw_parallelogram(0,
                       0,
                       floor_width,
                       floor_height,
                       slide * num_tiles,
                       Parallelogram.HORZ,
                       Direction.LEFT,
                       color="black",
                       global_x=global_x,
                       global_y=global_y)

    for width, height, xoff, yoff in [(0.8, 0.2, 0.85, 0.35),
                                      (0.3, 0.1, 0.2, 0.05)]:
        scaled_slide = (slide * num_tiles) / floor_height * height
        draw_cube(1,
                  1,
                  width,
                  height,
                  -0.5,
                  scaled_slide,
                  global_x=global_x + xoff,
                  global_y=global_y + yoff)
    draw_text(
        (global_x + (floor_width + slide * num_tiles) / 2, global_y - 0.6),
        "Bounding\nBoxes",
        font_size=label_text_size,
        line_height=label_text_size_line_height)


num_tiles = 5
floor_width = 1.5
floor_height = 0.6
slide = 0.1
cube_height = 1.5
circle_size = 0.05

num_matrix_entries = 4
matrix_width = 1.5
matrix_height = 2 / 5

dense_width = 2
coord_width = dense_width / 8

zero_fill_color = color_tuple_to_str((200, 200, 200))
non_zero_fill_color = "white"
font_size = 4

backbone_width = 3
backbone_height = 1.2

ssd_height = 1.5
ssd_width = 1.5
ssd_vert_offset = 0.3

label_text_size = 5
label_text_size_line_height = 0.2


#1.2
# 2.75 - 3.2
draw_rectangle(-0.2,
               -1,
               19.5,
               3.4)
draw_text((1.2, 2.1), "Shared Pipeline")
boxes_start = 8.4
hatching_color = color_tuple_to_str((220, 220, 220))

left_offset = 0.1
draw_rectangle(boxes_start,
               2.75,
               8.0,
               3.7,
               pattern_dir="north west lines",
               pattern_color=hatching_color)
draw_text((boxes_start + 1 + left_offset, 6.1), "PointPillars")
draw_rectangle(boxes_start, -3.4, 8.0, 2,
               pattern_dir="north east lines",
               pattern_color=hatching_color)
draw_text((boxes_start + 1.55 + left_offset, -3.1), "Sparse PointPillars")

draw_pointcloud(0, 0, num_tiles, floor_width, floor_height, slide, circle_size)

draw_arrow((floor_width + slide * num_tiles + 0.1, cube_height * 0.66),
           (floor_width + slide * num_tiles + 0.6, cube_height * 0.66))

fill_list = draw_pillarized_pointcloud(3, 0, num_tiles, floor_width,
                                       floor_height, slide, cube_height,
                                       circle_size)

draw_arrow((3 + floor_width + slide * num_tiles + 0.1, cube_height * 0.66),
           (3 + floor_width + slide * num_tiles + 0.6, cube_height * 0.66))

draw_gather_vectors(6, 0, num_matrix_entries, matrix_width, matrix_height,
                    coord_width, font_size)
draw_arrow((6 + floor_width + slide * num_tiles + 0.1, cube_height * 0.66),
           (6 + floor_width + slide * num_tiles + 0.6, cube_height * 0.66))

draw_vectorized_vectors(9, 0, num_matrix_entries, matrix_width, matrix_height,
                        coord_width, font_size)

draw_arrow(
    (9 + coord_width + matrix_width / 2, floor_height + cube_height + 0.1),
    (9 + coord_width + matrix_width / 2, floor_height + cube_height + 0.6))

draw_dense_pseudoimage(9, 3, num_tiles, dense_width, fill_list, font_size)

draw_arrow((9 + coord_width / 2 + dense_width + 0.2, 3 + dense_width / 2),
           (9 + coord_width / 2 + dense_width + 0.6, 3 + dense_width / 2))

draw_orig_backbone(12, 3 + dense_width / 2 - backbone_height / 2,
                   backbone_width, backbone_height)

draw_multi_point_arrow([
    (9 + coord_width + matrix_width / 2, -0.8),
    (9 + coord_width + matrix_width / 2, -3.5 + dense_width / 2),
    (9 + coord_width + dense_width + 0.6, -3.5 + dense_width / 2),
])

draw_sparse_backbone(12, -3.5 + dense_width / 2 - backbone_height / 2,
                     backbone_width, backbone_height)

draw_ssd(15, ssd_vert_offset, ssd_width, ssd_height)

draw_multi_point_arrow([
    (12 + backbone_width + 0.2,
     3 + dense_width / 2 - backbone_height / 2 + backbone_height / 2),
    (15 + ssd_width / 2,
     3 + dense_width / 2 - backbone_height / 2 + backbone_height / 2),
    (15 + ssd_width / 2, ssd_height + 0.2 + ssd_vert_offset),
])

draw_multi_point_arrow([
    (12 + backbone_width + 0.2,
     -3.5 + dense_width / 2 - backbone_height / 2 + backbone_height / 2),
    (15 + ssd_width / 2,
     -3.5 + dense_width / 2 - backbone_height / 2 + backbone_height / 2),
    (15 + ssd_width / 2, ssd_vert_offset - 0.2),
])

draw_arrow((15 + ssd_width + 0.2, cube_height * 0.66),
           (17, cube_height * 0.66))

draw_bounding_boxes(17, ssd_vert_offset, floor_width, floor_height, num_tiles,
                    slide)

file_str = pic.make()
# print(file_str)
with open('fig.tikz', 'w') as f:
    f.write(file_str)