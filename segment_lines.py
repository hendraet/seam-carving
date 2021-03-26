import argparse
from pathlib import Path
from typing import List, NoReturn, Tuple

import numpy
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageClass
from csaps import csaps
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.draw import line
from tqdm import tqdm, trange


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

    return maxima_locations


def convert_maxima_to_medial_seams(fully_connected_slice_maxima: List[List[Tuple]],
                                   slice_widths: List[int]) -> numpy.ndarray:
    medial_seams = []
    for maxima_group in fully_connected_slice_maxima:
        medial_seam = []
        for slice_idx, (maximum, slice_width) in enumerate(zip(maxima_group[:-1], slice_widths)):
            next_maximum = maxima_group[slice_idx + 1]

            half_slice_width = slice_width // 2
            x_start = sum(slice_widths[:slice_idx]) + half_slice_width + 1

            next_half_slice_width = slice_widths[slice_idx + 1] // 2
            missing_slice_width = slice_width - half_slice_width
            x_end = x_start + missing_slice_width + next_half_slice_width

            y_coords, x_coords = line(maximum[0], x_start, next_maximum[0], x_end)
            medial_seam += list(zip(y_coords[:-1], x_coords[:-1]))

        # since we always draw lines from the middle of the slice we need to add padding for the first and last slice
        first_slice_half = [(medial_seam[0][0], x) for x in range(medial_seam[0][1])]
        last_slice_half = [(medial_seam[-1][0], x) for x in range(medial_seam[-1][1] + 1, sum(slice_widths))]
        medial_seam = first_slice_half + medial_seam + last_slice_half
        medial_seams.append(medial_seam)

    return numpy.asarray(medial_seams)


def calculate_medial_seams(image: ImageClass, r: int = 8, b: float = 0.0003) -> numpy.ndarray:
    grayscale_image = image.convert("L")
    image_array = numpy.asarray(grayscale_image)
    sobel_image = ndimage.sobel(image_array)
    slices = numpy.array_split(sobel_image, r, axis=1)

    # Calculate maxima for each slice
    slice_maxima = []
    for image_slice in tqdm(slices, desc="Calculate seam maxima...", leave=False):
        maxima_locations = calculate_maxima_locations(image_slice, b)
        slice_maxima.append(maxima_locations)

    # Match maxima locations across slices to extract seams
    connected_slice_maxima = {}  # maps the end point of a line to the list of points that are part of this line
    for slice_idx in trange(r - 1, desc="Matching maxima...", leave=False):
        for left_maximum_idx, left_maximum in enumerate(slice_maxima[slice_idx]):
            right_maxima = slice_maxima[slice_idx + 1]
            if len(right_maxima) == 0:
                continue

            dists_left_to_right = numpy.absolute(left_maximum - right_maxima)
            min_dist_idx_right = numpy.argmin(dists_left_to_right)
            right_maximum = right_maxima[min_dist_idx_right]
            dists_right_to_left = numpy.absolute(right_maximum - slice_maxima[slice_idx])
            min_dist_idx_left = numpy.argmin(dists_right_to_left)

            if min_dist_idx_left == left_maximum_idx:
                start_point = (int(round(left_maximum)), slice_idx)
                end_point = (int(round(right_maximum)), slice_idx + 1)
                if start_point not in connected_slice_maxima.keys():
                    connected_slice_maxima[end_point] = [start_point, end_point]
                else:
                    connected_slice_maxima[end_point] = connected_slice_maxima[start_point] + [end_point]
                    connected_slice_maxima.pop(start_point)

    fully_connected_slice_maxima = [v for k, v in connected_slice_maxima.items() if len(v) == r]

    slice_widths = [image_slice.shape[1] for image_slice in slices]
    medial_seams = convert_maxima_to_medial_seams(fully_connected_slice_maxima, slice_widths)

    return medial_seams


def calculate_energy_map(original_image: ImageClass, sigma: float = 3.0) -> numpy.ndarray:
    grayscale_image = original_image.copy().convert("L")
    smoothed_image = gaussian_filter(numpy.asarray(grayscale_image), sigma=sigma, output=numpy.float64)

    # calculate the two addends of the formula independently but in a fast way
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

    # The calculation at the edges produces misleading results since one of the addends is always 0. Therefore, we set
    # these regions to 0, which corresponds to the energy of background pixels.
    energy_map[0, :] = 0
    energy_map[-1, :] = 0
    energy_map[:, 0] = 0
    energy_map[:, -1] = 0

    return energy_map


def get_local_energy_map(medial_seam: numpy.ndarray, next_medial_seam: numpy.ndarray,
                         energy_map: numpy.ndarray) -> numpy.ndarray:
    width = medial_seam.shape[0]

    i_start = medial_seam[:, 0].min()
    absolute_height_diff = next_medial_seam[:, 0].max() - i_start
    i_end = i_start + absolute_height_diff
    real_local_energy_map = energy_map[i_start:i_end + 1, :].copy()

    # The points between two medial seems rarely form a 2D array. Thus, we create a 2D that contains all relevant
    # values from the energy map and set all irrelevant pixel to the sum of all values in the energy map (called
    # "empty value"). Since we try to minimise a sum of pixels and values in the energy are always > 0 this
    # should be sufficient so that those "empty values" are never part of an actual solution.
    empty_value = numpy.sum(real_local_energy_map)
    mask = numpy.linspace([i_start] * width, [i_end] * width, num=absolute_height_diff + 1)
    mask = numpy.where(
        numpy.logical_or(
            mask < medial_seam[:, 0],
            mask > next_medial_seam[:, 0]
        ), False, True)

    assert real_local_energy_map.shape == mask.shape
    local_energy_map = numpy.where(mask, real_local_energy_map, empty_value)

    return local_energy_map


def calculate_minimum_energy_map(medial_seam: numpy.ndarray, local_energy_map: numpy.ndarray) -> numpy.ndarray:
    # use DP to calculate the optimal energy paths for each starting pixel
    width = medial_seam.shape[0]
    height = local_energy_map.shape[0]
    minimum_energy_map = numpy.zeros_like(local_energy_map)
    minimum_energy_map[:, 0] = local_energy_map[:, 0]

    for j in trange(1, width, desc="Calculating minimum energy map...", leave=False):
        for i in range(height):
            previous_energies = [minimum_energy_map[i, j - 1]]
            if i - 1 >= 0:
                previous_energies += [minimum_energy_map[i - 1, j - 1]]
            if i + 1 < height:
                previous_energies += [minimum_energy_map[i + 1, j - 1]]
            minimum_energy_map[i, j] = local_energy_map[i, j] + min(previous_energies)

    return minimum_energy_map


def get_optimal_separating_seam(medial_seam: numpy.ndarray, local_energy_map: numpy.ndarray,
                                minimum_energy_map: numpy.ndarray) -> List[Tuple]:
    width = medial_seam.shape[0]
    height = local_energy_map.shape[0]
    i_start = medial_seam[:, 0].min()
    current_i = numpy.argmin(minimum_energy_map[:, -1])
    optimal_seam = [(current_i, width - 1)]

    for j in reversed(range(0, width - 1)):
        best_i = current_i
        best_energy = minimum_energy_map[current_i, j]
        if current_i - 1 >= 0 and minimum_energy_map[current_i - 1, j] < best_energy:
            best_i = current_i - 1
            best_energy = minimum_energy_map[best_i, j]
        if current_i + 1 < height and minimum_energy_map[current_i + 1, j] < best_energy:
            best_i = current_i + 1
        optimal_seam.append((best_i, j))
        current_i = best_i

    absolute_optimal_seam = [(i + i_start, j) for i, j in reversed(optimal_seam)]
    return absolute_optimal_seam


def calculate_separating_seams(medial_seams: numpy.ndarray, energy_map: numpy.ndarray) -> List:
    separating_seams = []
    for seam_idx, medial_seam in enumerate(tqdm(medial_seams[:-1], desc="Calculating separating seams...",
                                                leave=False)):
        next_medial_seam = medial_seams[seam_idx + 1]
        local_energy_map = get_local_energy_map(medial_seam, next_medial_seam, energy_map)
        minimum_energy_map = calculate_minimum_energy_map(medial_seam, local_energy_map)
        optimal_seperating_seam = get_optimal_separating_seam(medial_seam, local_energy_map, minimum_energy_map)
        separating_seams.append(optimal_seperating_seam)

    return separating_seams


def main(args: argparse.Namespace) -> NoReturn:
    for image_path in tqdm(args.input_images, desc="Processing images...", leave=False):
        input_image = Image.open(image_path)
        slice_width = input_image.width // args.r

        # A. Medial Seam Computation
        medial_seams = calculate_medial_seams(input_image, r=args.r, b=args.b)

        # B. Separating Seam Computation
        energy_map = calculate_energy_map(input_image, sigma=args.sigma)
        separating_seams = calculate_separating_seams(medial_seams, energy_map)

        base_output_filename = args.output_dir / f"{image_path.stem}"
        output_filename = f"{base_output_filename}_lines{image_path.suffix}"
        output_image = ImageDraw.Draw(input_image)
        if args.debug:
            output_filename = f"{base_output_filename}_lines_debug_r{args.r}_b{args.b}{image_path.suffix}"
            for i in range(args.r - 1):
                border_x = (i + 1) * slice_width
                patch_border = [(border_x, 0), (border_x, input_image.height - 1)]
                output_image.line(patch_border, fill=(255, 0, 255), width=3)
            for medial_seam in medial_seams:
                points = [(x, y) for y, x in medial_seam]
                output_image.line(points, fill=(0, 255, 0), width=5)

        for separating_seam in separating_seams:
            points = [(x, y) for y, x in separating_seam]
            output_image.line(points, fill=(255, 0, 0), width=5)

        input_image.save(output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="segment a given image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_images", type=Path, nargs="+",
                        help="Filename(s) of the image(s) that should be processed")
    parser.add_argument("--output-dir", type=Path, default="images",
                        help="The directory in which the resulting image(s) should be saved")
    parser.add_argument("--debug", action="store_true", default=False, help="Display additional information")
    parser.add_argument("--r", type=int, default=8, help="Hyperparameter r")
    parser.add_argument("--b", type=float, default=0.0003, help="Hyperparameter b")
    parser.add_argument("--sigma", type=float, default=3.0, help="Hyperparameter sigma")
    main(parser.parse_args())
