import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize


def find_edge_points(cloud, threshold=1):
    edge_points = []
    for i in range(len(cloud)):
        if 0 < i < len(cloud) - 1:
            prev_diff = np.linalg.norm(cloud[i - 1] - cloud[i])
            next_diff = np.linalg.norm(cloud[i + 1] - cloud[i])
            if prev_diff > threshold and next_diff > threshold:
                edge_points.append(cloud[i])
    return np.array(edge_points)


def distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


def minimize_objective_function(objective_function, source_cloud, target_cloud, initial_params):
    result = minimize(
        objective_function,
        initial_params,
        args=(source_cloud, target_cloud),
        method='BFGS'
    )

    print("Wynik optymalizacji:", result.x)

    params = result.x
    transformed_cloud = apply_transformation(source_cloud, params)

    return transformed_cloud, target_cloud


def apply_transformation(source_cloud, params):
    tx, ty, theta = params
    transformed_cloud = np.dot(source_cloud, np.array([[np.cos(theta), -np.sin(theta)],
                                                       [np.sin(theta), np.cos(theta)]]))
    transformed_cloud += np.array([tx, ty])
    return transformed_cloud


def objective_function_spin(params, source_cloud, target_cloud):
    transformed_source_cloud = apply_transformation(source_cloud, params)

    kdtree = KDTree(target_cloud)

    _, indices = kdtree.query(transformed_source_cloud)

    total_distance = np.sum(
        [distance(transformed_source_cloud[i], target_cloud[indices[i]]) for i in range(len(transformed_source_cloud))])

    return total_distance


def calculate_rmse(transformed_cloud, target_cloud):
    kdtree = KDTree(target_cloud)
    _, indices = kdtree.query(transformed_cloud)
    distances = np.linalg.norm(transformed_cloud - target_cloud[indices], axis=1)
    rmse = np.sqrt(np.mean(distances**2))
    return rmse


def filter_points_by_distance(points, max_distance):
    filtered_points = []

    for i in range(0, len(points)-1):
        ok_x = abs(points[i, 0] - points[i + 1, 0])
        ok_y = abs(points[i, 1] - points[i + 1, 1])
        if ok_x < max_distance and ok_y < max_distance:
            filtered_points.append([points[i, 0], points[i, 1]])

    return np.array(filtered_points)


def icp_algorithm(cloud1_name, cloud2_name):
    source_cloud = np.loadtxt(f'Slices/{cloud1_name}', delimiter=',')
    target_cloud = np.loadtxt(f'Slices/{cloud2_name}', delimiter=',')

    source_cloud = filter_points_by_distance(source_cloud, 1000)

    edge_source_cloud = find_edge_points(source_cloud)
    edge_target_cloud = find_edge_points(target_cloud)

    init = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

    best_cloud = None
    best_rmse = np.inf

    for i in init:
        for j in init:
            initial_params = np.array([j, 0.0, i])
            transformed_source_cloud, target_cloud = minimize_objective_function(
                objective_function_spin, edge_source_cloud, edge_target_cloud, initial_params)

            rmse = calculate_rmse(transformed_source_cloud, target_cloud)
            if rmse < best_rmse:
                best_rmse = rmse
                best_cloud = transformed_source_cloud

    return best_cloud, target_cloud, best_rmse
