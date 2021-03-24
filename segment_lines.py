import argparse
import os
from pathlib import Path
from typing import List, NoReturn

import numpy
from tqdm import tqdm, trange
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageClass
from csaps import csaps
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def calculate_maxima_locations(image_slice: numpy.ndarray, b: float) -> numpy.ndarray:
    slice_height, slice_width = image_slice.shape
    projection_profile = numpy.zeros((slice_height,))
    for i in range(slice_height):
        projection_profile[i] = numpy.sum(image_slice[i])
    x = numpy.linspace(0, len(projection_profile) - 1, num=len(projection_profile))
    smoothed_spline = csaps(x, projection_profile, smooth=b).spline
    d1 = smoothed_spline.derivative(nu=1)
    d2 = smoothed_spline.derivative(nu=2)
    extrema = d1.roots()
    maxima_locations = extrema[d2(extrema) < 0.0]

    # TODO: remove
    # import matplotlib.pyplot as plt
    # _, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    # smoothed_projection_profile = csaps(x, projection_profile, xi, smooth=b)
    # ax1.plot(x, projection_profile, "o", xi, smoothed_projection_profile, "-")
    # ax2.plot(xi, smoothed_spline(xi), "-", maxima, smoothed_spline(maxima), "o")
    # plt.savefig("tests/testdata/diva_line_segmentation/segment_test/smoothed_pro_pro.png")

    return maxima_locations


def calculate_line_seams(image: ImageClass, r: int = 8, b: float = 0.0003) -> numpy.ndarray:  # TODO: tune params
    grayscale_image = image.convert("L")
    image_array = numpy.asarray(grayscale_image)
    sobel_image = ndimage.sobel(image_array)  # TODO: use already binarised image and apply otsu threshold
    # Image.fromarray(sobel_image).save("tests/testdata/diva_line_segmentation/segment_test/sobel.png")
    slices = numpy.array_split(sobel_image, r, axis=1)

    # Calculate maxima for each slice
    slice_maxima = []
    for image_slice in tqdm(slices, desc="Calculate seam maxima...", leave=False):
        maxima_locations = calculate_maxima_locations(image_slice, b)
        slice_maxima.append(maxima_locations)

    # Match maxima locations across slices to extract seams
    # TODO: maybe function
    lines = {}  # maps the end point of a line to the list of points that are part of this line
    for slice_idx in trange(r - 1, desc="Matching maxima...", leave=False):
        for left_maximum_idx, left_maximum in enumerate(slice_maxima[slice_idx]):
            right_maxima = slice_maxima[slice_idx + 1]
            if len(right_maxima) == 0:
                continue  # TODO: properly handle empty maxima arrays [also if they are on the left

            dists_left_to_right = numpy.absolute(left_maximum - right_maxima)
            min_dist_idx_right = numpy.argmin(dists_left_to_right)
            right_maximum = right_maxima[min_dist_idx_right]
            dists_right_to_left = numpy.absolute(right_maximum - slice_maxima[slice_idx])
            min_dist_idx_left = numpy.argmin(dists_right_to_left)

            if min_dist_idx_left == left_maximum_idx:
                # print(f"({left_maximum_idx}, {left_maximum}), ({min_dist_idx_right}, {right_maximum})")
                start_point = (slice_idx, int(round(left_maximum)))
                end_point = (slice_idx + 1, int(round(right_maximum)))
                if start_point not in lines.keys():
                    lines[end_point] = [start_point, end_point]
                else:
                    lines[end_point] = lines[start_point] + [end_point]
                    lines.pop(start_point)

    # TODO: maybe merge lines somehow
    seams = [v for k, v in lines.items() if len(v) == r]

    return numpy.asarray(seams)


def calculate_energy_map(original_image: ImageClass, sigma: float = 3.0) -> numpy.ndarray:
    grayscale_image = original_image.copy().convert("L")
    smoothed_image = gaussian_filter(numpy.asarray(grayscale_image), sigma=sigma, output=numpy.float64)

    # calculate the two summands of the formula independently but in a fast way
    # the arrays have to be padded with 2 rows/column because they are shifted by 1 and need additional padding to
    # handle the calculation at the edges, e.g. that the value at pixel_value[j - 1] = 0 if j - 1 < 0
    vertical_padding = numpy.zeros((grayscale_image.height, 2))
    j_plus_one = numpy.concatenate((smoothed_image, vertical_padding), axis=1)
    j_minus_one = numpy.concatenate((vertical_padding, smoothed_image), axis=1)
    vertical_energy_map = numpy.absolute((j_plus_one - j_minus_one) / 2)[:, 1:-1]

    horizontal_padding = numpy.zeros((2, grayscale_image.width))
    i_plus_one = numpy.concatenate((smoothed_image, horizontal_padding), axis=0)
    i_minus_one = numpy.concatenate((horizontal_padding, smoothed_image), axis=0)
    horizontal_energy_map = numpy.absolute((i_plus_one - i_minus_one) / 2)[1:-1, :]
    energy_map = vertical_energy_map + horizontal_energy_map

    return energy_map


def calculate_seams(line_seams: numpy.ndarray, energy_map: numpy.ndarray, r: int) -> List:
    for i, line_seam in enumerate(line_seams[:-1]):
        next_line_seam = line_seams[i + 1]
        line_seam_height_diffs = [(b[1] - a[1]) for a, b in zip(line_seam, next_line_seam)]
        if any([dist < 0 for dist in line_seam_height_diffs]):
            print(f"Lines {i} and {i + 1} are intersecting. Skipping...")
            continue

        i_start = line_seam[:, 1].min()
        absolute_height_diff = next_line_seam[:, 1].max() - i_start
        i_end = i_start + absolute_height_diff
        local_energy_map = energy_map[i_start:i_end, :].copy()
        # TODO: next step: fill values that are outsied of lines with high (?) values
        print()

    return []  # TODO


def main(args: argparse.Namespace) -> NoReturn:
    # TODO: impl image argument and remove this
    input_images = [
        "test_images/00000020.jpg",
        "test_images/00000048.jpg",
        "test_images/00000055.jpg",
        "test_images/00000062.jpg",
        "test_images/069a.jpg",
        "test_images/083a.jpg"
    ]
    input_images = [Path(path) for path in input_images]

    for image_path in tqdm(input_images, desc="Processing images...", leave=False):
        input_image = Image.open(image_path)

        # TODO: group line_calculation in function
        # A. Medial Seam Computation
        r = 8
        line_seams = calculate_line_seams(input_image, r=r)

        # B. Separating Seam Computation
        energy_map = calculate_energy_map(input_image)
        seams = calculate_seams(line_seams, energy_map, r)

        base_output_filename = args.output_dir / f"{image_path.stem}"
        if args.debug:
            # visualize lines
            slice_width = input_image.width // r
            half_slice_width = slice_width // 2
            image_with_lines = ImageDraw.Draw(input_image)
            for line in line_seams:
                corrected_line = [((p[0]) * slice_width + half_slice_width, p[1]) for p in line]
                image_with_lines.line(corrected_line, fill=(0, 255, 0), width=5)
            for i in range(r - 1):
                border_x = (i + 1) * slice_width
                patch_border = [(border_x, 0), (border_x, input_image.height - 1)]
                image_with_lines.line(patch_border, fill=(255, 0, 255), width=3)

            output_filename = f"{base_output_filename}_lines{image_path.suffix}"
            input_image.save(output_filename)


if __name__ == '__main__':
    # TODO: remove code from other repo
    # TODO: make public

    parser = argparse.ArgumentParser(
        description="segment a given image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default="images")
    parser.add_argument("--debug", action="store_true", default=False)
    main(parser.parse_args())
