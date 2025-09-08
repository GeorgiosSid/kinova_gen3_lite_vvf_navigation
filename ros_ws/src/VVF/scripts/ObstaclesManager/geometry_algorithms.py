from sklearn.decomposition import PCA
from scipy.optimize import minimize
import numpy as np

def fit_cylinder_PCA(points):
    # Principal axis using PCA
    pca = PCA(n_components=2)  # Fit the principal axis
    pca.fit(points)
    axis_vector = pca.components_[0]  # Principal axis

    # Project points onto the axis and find min and max projections
    projections = points @ axis_vector
    min_projection_idx, max_projection_idx = projections.argmin(), projections.argmax()
    bottom_point, top_point = points[min_projection_idx], points[max_projection_idx]

    # Compute radius as the maximum distance from the axis
    distances = np.linalg.norm(np.cross(points - bottom_point, axis_vector), axis=1)
    radius = distances.max()

    return bottom_point, top_point, radius

def fit_cylinder_BFGS(points):
    def cylinder_loss(params):
        x0, y0, z0, nx, ny, nz, r = params
        axis = np.array([nx, ny, nz])
        axis /= np.linalg.norm(axis)  # Normalize the axis direction

        # Compute distances to the cylinder axis
        relative_positions = points - np.array([x0, y0, z0])
        projections = relative_positions @ axis
        closest_points = relative_positions - projections[:, None] * axis
        distances = np.linalg.norm(closest_points, axis=1)

        # Loss: variance of distances around the radius
        return np.mean((distances - r) ** 2)

    # Initial guess: center at mean of points, axis along Z, radius as std deviation
    centroid = np.mean(points, axis=0)
    initial_axis = np.array([0, 0, 1])  # Default cylinder axis along Z
    r_guess = np.std(points[:, :3])
    initial_guess = [*centroid, *initial_axis, r_guess]

    result = minimize(cylinder_loss, initial_guess, method='BFGS')
    params = result.x

    # Extract optimized parameters
    x0, y0, z0, nx, ny, nz, r = params
    axis = np.array([nx, ny, nz])
    axis /= np.linalg.norm(axis)  # Normalize axis

    # Compute projections along the axis
    relative_positions = points - np.array([x0, y0, z0])
    projections = relative_positions @ axis

    # Find the min and max projections to determine the top and bottom points
    min_proj = projections.min()
    max_proj = projections.max()
    bottom_point = np.array([x0, y0, z0]) + min_proj * axis
    top_point = np.array([x0, y0, z0]) + max_proj * axis

    return bottom_point, top_point, r

def fit_sphere(points):
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    return center, radius

def best_fit(cylinder_alg, points):
    """
    Determines the best fit (cylinder or sphere) for a set of points,
    based on minimum volume.

    Args:
        cylinder_alg (str): The algorithm to use for cylinder fitting ('PCA' or 'BFGS').
        points (np.ndarray): Nx3 array of points.

    Returns:
        dict: A dictionary containing the type of best fit ('cylinder' or 'sphere'),
              and corresponding parameters:
              - For 'cylinder': 'bottom_point', 'top_point', 'radius'
              - For 'sphere': 'center', 'radius'
    """
    # Fit a cylinder
    if cylinder_alg == "PCA":
        bottom_point, top_point, radius = fit_cylinder_PCA(points)
    elif cylinder_alg == "BFGS":
        bottom_point, top_point, radius = fit_cylinder_BFGS(points)
    else:
        raise ValueError("Invalid cylinder algorithm. Use 'PCA' or 'BFGS'.")

    # Compute the cylinder height and volume
    cylinder_height = np.linalg.norm(top_point - bottom_point)
    cylinder_volume = np.pi * radius**2 * cylinder_height

    # Fit a sphere
    sphere_center, sphere_radius = fit_sphere(points)
    sphere_volume = (4 / 3) * np.pi * sphere_radius**3

    # Determine the best fit
    if sphere_volume < cylinder_volume:
        return {
            "type": "sphere",
            "center": sphere_center,
            "radius": sphere_radius
        }
    else:
        return {
            "type": "cylinder",
            "bottom_point": bottom_point,
            "top_point": top_point,
            "radius": radius
        }
