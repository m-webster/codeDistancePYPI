import stim
from collections.abc import Callable


def _filter_flattened_dem(
    dem: stim.DetectorErrorModel,
    keep_error_func: Callable[[stim.DemInstruction], bool],
    filter_detector_instructions: bool = False,
) -> stim.DetectorErrorModel:
    """Filters a flattened detector error model."""
    if not filter_detector_instructions:
        filtered_dem = stim.DetectorErrorModel()
        for instruction in dem:
            if instruction.type == "error":
                if keep_error_func(instruction):
                    filtered_dem.append(instruction)
            else:
                filtered_dem.append(instruction)
        return filtered_dem

    # First, filter the error instructions and identify kept detectors
    filtered_errors = []
    kept_detectors = set()
    for instruction in dem:
        if instruction.type == "error":
            if keep_error_func(instruction):
                filtered_errors.append(instruction)
                for target in instruction.targets_copy():
                    if target.is_relative_detector_id():
                        kept_detectors.add(target.val)

    # Re-build the DEM, keeping only the relevant detector instructions
    final_filtered_dem = stim.DetectorErrorModel()
    for instruction in dem:
        if instruction.type == "detector":
            # Keep detector instruction if any of its targets are in kept_detectors
            if any(
                t.val in kept_detectors
                for t in instruction.targets_copy()
                if t.is_relative_detector_id()
            ):
                final_filtered_dem.append(instruction)
        elif instruction.type != "error":
            final_filtered_dem.append(instruction)

    for error in filtered_errors:
        final_filtered_dem.append(error)

    return final_filtered_dem


def filter_dem_errors_by_detector_basis(
    dem: stim.DetectorErrorModel,
    detector_basis: Callable[[int], str],
    desired_basis: str,
    filter_detector_instructions: bool = False,
) -> stim.DetectorErrorModel:
    """Filters a DEM, keeping only errors where all detectors match a desired basis.

    This function first flattens the DEM, which removes all REPEAT blocks and
    resolves all relative detector indices to absolute ones.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to filter.
    detector_basis : Callable[[int], str]
        A function that returns the basis ('X' or 'Z') of a given absolute
        detector index.
    desired_basis : str
        The basis to keep ('X' or 'Z').
    filter_detector_instructions : bool, optional
        If True, also filters out `detector` instructions for detectors that
        are no longer present in any error. Defaults to False.

    Returns
    -------
    stim.DetectorErrorModel
        A new DEM containing only the errors whose detectors are all of the
        `desired_basis`.
    """
    flattened_dem = dem.flattened()

    def keep_error_func(instruction: stim.DemInstruction) -> bool:
        detector_targets = [
            t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
        ]
        if not detector_targets:
            # Keep errors with no detectors (e.g. pure logical errors)
            return True
        return all(detector_basis(t) == desired_basis for t in detector_targets)

    return _filter_flattened_dem(
        flattened_dem, keep_error_func, filter_detector_instructions
    )


def _dem_has_observable_flip(dem: stim.DetectorErrorModel) -> bool:
    """Checks if any error in the DEM flips a logical observable."""
    for instruction in dem.flattened():
        if instruction.type == "error":
            if any(t.is_logical_observable_id() for t in instruction.targets_copy()):
                return True
    return False


def filter_dem_to_one_basis(
    dem: stim.DetectorErrorModel, detector_basis: Callable[[int], str]
) -> stim.DetectorErrorModel:
    """Filters a DEM to a single basis, ensuring it's the one with a logical error.

    This function separates a DEM into its X-basis and Z-basis components. It
    succeeds only if exactly one of these components contains an error that
    flips a logical observable.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to filter.
    detector_basis : Callable[[int], str]
        A function that returns the basis ('X' or 'Z') of a given absolute
        detector index.

    Returns
    -------
    stim.DetectorErrorModel
        The filtered DEM for the single basis that has a logical error.

    Raises
    ------
    ValueError
        If both the X and Z filtered DEMs have logical errors, or if neither of
        them do.
    """
    dem_x = filter_dem_errors_by_detector_basis(dem, detector_basis, "X")
    dem_z = filter_dem_errors_by_detector_basis(dem, detector_basis, "Z")

    x_has_flip = _dem_has_observable_flip(dem_x)
    z_has_flip = _dem_has_observable_flip(dem_z)

    if x_has_flip and not z_has_flip:
        return dem_x
    if z_has_flip and not x_has_flip:
        return dem_z
    if x_has_flip and z_has_flip:
        raise ValueError(
            "Both X and Z filtered DEMs have logical errors. The basis is ambiguous."
        )
    raise ValueError(
        "Neither X nor Z filtered DEMs have a logical error. Cannot determine basis."
    )


def filter_by_det_basis_using_chromobius_coords(
    dem: stim.DetectorErrorModel,
) -> stim.DetectorErrorModel:
    """Filters a DEM with Chromobius-style coordinates to a single logical basis.

    This function uses the 4th coordinate of each detector to determine its
    basis according to the Chromobius convention:
        - 0, 1, 2: X-basis
        - 3, 4, 5: Z-basis
        - -1 or other: Ignored (neither X nor Z)

    It then filters the DEM to keep only the errors that trigger detectors only 
    from the basis ('X' or 'Z') of the logical observable.
    
    More concretely, for each of basis=X and basis=Z it filters to a DEM that
    only keeps an error instruction such as `error(p) Di Dj Dk ...` if
    *all* the detectors that the error triggers are of the desired basis.
    A filtered DEM is created for both basis=X and basis=Z. For a CSS circuit
    that has a logical observable declared in one basis, only one of these
    filtered DEMs will contain any errors that trigger an observable. It 
    is this DEM that is returned.
    
    For a CSS circuit containing only CNOT gates, the filtered DEM will only 
    contain errors that are of the opposite basis to the observable (X errors
    if the observable is Z, for example). However error bases are classified by
    the basis of detector they trigger, so the code correctly handles circuits
    transpiled to CZ gates and Hadamards (such as SI1000 noise models).
    

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model, with detectors annotated with 4D coordinates.

    Returns
    -------
    stim.DetectorErrorModel
        The filtered DEM for the single basis that has a logical error.
    """
    coords = dem.get_detector_coordinates()

    def detector_basis(detector_index: int) -> str:
        """Determines detector basis from 4th coordinate."""
        if detector_index not in coords:
            return "Unknown"
        det_coords = coords[detector_index]
        if len(det_coords) < 4:
            return "Unknown"
        basis_val = det_coords[3]
        if 0 <= basis_val <= 2:
            return "X"
        if 3 <= basis_val <= 5:
            return "Z"
        return "Unknown"

    return filter_dem_to_one_basis(dem, detector_basis)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python dem_detector_filtering.py <circuits_dir>")
        sys.exit(1)

    circuits_dir = Path(sys.argv[1])

    header = f"{'Circuit File':<80} {'Property':<20} {'Original':>10} {'Filtered':>10}"
    print(header)
    print("-" * len(header))

    for circuit_path in sorted(circuits_dir.glob("*.stim")):
        circuit = stim.Circuit.from_file(circuit_path)
        original_dem = circuit.detector_error_model()

        try:
            filtered_dem = filter_by_det_basis_using_chromobius_coords(original_dem)
        except ValueError as e:
            print(f"Skipping {circuit_path.name}: {e}")
            continue

        try:
            original_dist = len(original_dem.shortest_graphlike_error())
        except (ValueError, IndexError):
            original_dist = "Error"

        try:
            filtered_dist = len(filtered_dem.shortest_graphlike_error())
        except (ValueError, IndexError):
            filtered_dist = "Error"

        print(
            f"{circuit_path.name:<80} {'num_detectors':<20} {original_dem.num_detectors:>10} {filtered_dem.num_detectors:>10}"
        )
        print(
            f"{'':<80} {'num_errors':<20} {original_dem.num_errors:>10} {filtered_dem.num_errors:>10}"
        )
        print(
            f"{'':<80} {'distance':<20} {str(original_dist):>10} {str(filtered_dist):>10}"
        )
        print("-" * len(header))
