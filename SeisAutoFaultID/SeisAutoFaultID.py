# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:33:04 2024

@author: Xiaoming Zhang
"""
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')  # Clear all variables before the script begins to run.

# Required Imports
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.colors import qualitative
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import re
import glob

# Set working directory
os.chdir(
    'C:/Users/Xiaoming Zhang/Desktop/postdoc_Xiaoming Zhang/'
    'NoamWork/Forge_Noam_project/UtahFORGE_allData/fittingFaults'
)

# Conversion factor
USFT_TO_M = 0.304800609601219

# Well class to encapsulate well properties and processing
class Well:
    def __init__(self, name, file_path, start_row, columns, column_names, plot_color):
        self.name = name
        self.file_path = file_path
        self.start_row = start_row
        self.columns = columns
        self.column_names = column_names
        self.plot_color = plot_color
        self.data = self._read_excel_data()
        self.depth = self._convert_to_meters(self.data['True Vertical Distance'])
        self.northing = self._convert_to_meters(self.data['Northing'])
        self.easting = self._convert_to_meters(self.data['Easting'])
        self.horizontal_distance = self._calculate_horizontal_distance()

    def to_dict(self):
        """Generate a dictionary with well properties."""
        return {
            "depth_16a": self.depth,
            "northing_16a": self.northing,
            "easting_16a": self.easting,
            "horizontal_16a": self.horizontal_distance
        }

    def _read_excel_data(self):
        """
        Reads specific columns from an Excel file starting from a specific row.
        """
        df = pd.read_excel(
            self.file_path,
            engine='openpyxl',
            header=None,
            skiprows=self.start_row,
            usecols=self.columns
        )
        df.columns = self.column_names
        return df

    def _convert_to_meters(self, column):
        """
        Converts a column from US feet to meters.
        """
        return USFT_TO_M * column

    def _calculate_horizontal_distance(self):
        """
        Calculates the horizontal distance using the difference between northing and easting coordinates.
        """
        return np.sqrt((self.northing - self.northing.iloc[0]) ** 2 + (self.easting - self.easting.iloc[0]) ** 2)

    def plot_trajectory_plotly(self, fig):
        """Add well trajectory to a Plotly 3D figure."""
        fig.add_trace(go.Scatter3d(
            x=self.easting,
            y=self.northing,
            z=self.depth,
            mode='lines',
            line=dict(color=self.plot_color, width=4),
            name=self.name
        ))

    def plot_trajectory_matplotlib(self, ax):
        """Add well trajectory to a Matplotlib 3D plot."""
        ax.plot(self.easting, self.northing, self.depth, color=self.plot_color, linewidth=2, label=self.name)

def fit_2d_rectangles_to_3d_points(points, eps=1.5, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)
    rectangles, rect_info = [], []

    for label in unique_labels:
        cluster_points = points[labels == label]
        pca = PCA(n_components=2)
        pca.fit(cluster_points)
        center = np.mean(cluster_points, axis=0)
        axes = pca.components_
        lengths = 2 * np.max(np.abs((cluster_points - center) @ axes.T), axis=0)
        corners = np.array([
            center + 0.5 * dx * lengths[0] * axes[0] + 0.5 * dy * lengths[1] * axes[1]
            for dx in [-1, 1] for dy in [-1, 1]
        ])
        rectangles.append(corners)
        rect_info.append({
            "center": center,
            "length": lengths[0],
            "width": lengths[1],
            "num_points": len(cluster_points),
            "axes": axes
        })

    return rectangles, labels, rect_info

def fit_2d_rectangles_to_3d_points_labelled(points):    
    labels = points['label']
    unique_labels = set(labels)
    unique_labels.discard(-1)
    rectangles, rect_info = [], []

    for label in unique_labels:
        cluster_data = points[labels == label]
        pca = PCA(n_components=2)
        cluster_points = cluster_data[['X (m)', 'Y (m)', 'depth (m)']].values
        pca.fit(cluster_points)
        center = np.mean(cluster_points, axis=0)
        axes = pca.components_
        lengths = 2 * np.max(np.abs((cluster_points - center) @ axes.T), axis=0)
        corners = np.array([
            center + 0.5 * dx * lengths[0] * axes[0] + 0.5 * dy * lengths[1] * axes[1]
            for dx in [-1, 1] for dy in [-1, 1]
        ])
        rectangles.append(corners)
        rect_info.append({
            "center": center,
            "length": lengths[0],
            "width": lengths[1],
            "num_points": len(cluster_points),
            "axes": axes
        })

    return rectangles, labels, rect_info

def project_points_onto_plane(points, labels, rect_info):
    """
    Project 3D points onto a plane defined by its center and axes.
    Args:
        points: Array of 3D points to project.
        rect_info: dict, the rectangle's properties including center, axes, length, and width.
    Returns:
        projected_points: Points projected onto the plane (in 3D).
        local_coordinates: Points in the 2D coordinate system of the plane.
    """
    projected_points = []
    local_coordinates = []    
    for i in range(len(points)):
        label = labels[i]
        point = points[i]
        if labels[i] > -1:
            rect_center = rect_info[label]["center"]
            rect_axes = rect_info[label]["axes"]
    
            normal = np.cross(rect_axes[0], rect_axes[1])  # Compute the normal vector of the plane
            normal /= np.linalg.norm(normal)    # Normalize the normal vector
    
            # Compute the projection of point onto the plane
            to_center = point - rect_center
            distance_to_plane = np.dot(to_center, normal) # Perpendicular distance to the plane
            projections = point - distance_to_plane * normal  # Projection in 3D space 
            # Convert projected points to local 2D coordinates (w.r.t. plane axes)
            local_coords = np.dot(projections - rect_center, rect_axes.T)  # 2D coordinates on the plane
        else:
            projections = point 
            local_coords = point[:2] # 2D coordinates on the plane
        
        projected_points.append(projections)
        local_coordinates.append(local_coords)
        
    projected_points = np.vstack(projected_points)
    local_coordinates = np.vstack(local_coordinates)
    
    return projected_points, local_coordinates

def check_point_in_rectangle(point, rect_info, distance_limit, length_coeffi = 2.):
    """
    Check if a point lies inside the rectangle based on both projection and distance from the rectangle's plane.

    Parameters:
    - point: ndarray, the coordinates of the point (x, y, z).
    - rect_info: dict, the rectangle's properties including center, axes, length, and width.
    - distance_limit: float, the maximum allowable distance from the rectangle's plane.

    Returns:
    - bool: True if the point belongs to the rectangle, False otherwise.
    """
    center = rect_info["center"]
    axes = rect_info["axes"]
    lengths = np.array([rect_info["length"], rect_info["width"]])

    # Compute the normal vector of the rectangle's plane
    normal = np.cross(axes[0], axes[1])

    # Calculate the distance of the point to the plane
    relative_position = point - center
    distance_to_plane = np.abs(np.dot(relative_position, normal)) / np.linalg.norm(normal)

    # If the point is too far from the plane, return False
    if distance_to_plane > distance_limit:
        return False

    # Project the point onto the rectangle's axes
    projections = np.dot(relative_position, axes.T)

    # Check if the projections lie within the rectangle's bounds
    return np.all(np.abs(projections) <= 0.5 * length_coeffi *lengths)

def update_cluster_and_rectangle(cluster_points, new_points):
    """Update a cluster and its fitted rectangle with new points."""
    updated_points = np.vstack([cluster_points, new_points])

    # Recalculate the PCA and rectangle
    pca = PCA(n_components=2)
    pca.fit(updated_points)
    center = np.mean(updated_points, axis=0)
    axes = pca.components_
    lengths = 2 * np.max(np.abs((updated_points - center) @ axes.T), axis=0)
    corners = np.array([
        center + 0.5 * dx * lengths[0] * axes[0] + 0.5 * dy * lengths[1] * axes[1]
        for dx in [-1, 1] for dy in [-1, 1]
    ])
    rect_info = {
        "center": center,
        "length": lengths[0],
        "width": lengths[1],
        "num_points": len(updated_points),
        "axes": axes
    }
    return updated_points, corners, rect_info

def iterative_clustering_and_update(points, batch_size, distance_limit):
    """Iteratively classify points and update clusters and rectangles."""
    rectangles, labels, rect_info = fit_2d_rectangles_to_3d_points_labelled(points)
    
    remaining_data = points[labels == -1]  # Unclustered points
    
    # print("Before len(remaining_data) = ", len(remaining_data))
    
    remaining_index = np.where(labels == -1)[0]
    # print(remaining_labelIndex)
    remaining_points = remaining_data[['X (m)', 'Y (m)', 'depth (m)']].values

    # # Start by processing remaining points in batches
    num_batches = len(remaining_points) // batch_size + (1 if len(remaining_points) % batch_size > 0 else 0)  # Handle cases where remaining_points is not divisible by 10
    
    for batch_idx in range(num_batches):
        # Get the current batch of points (handle case where batch may have fewer than batch_size points)
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(remaining_points))
        current_batch = remaining_points[start_idx:end_idx]
        index_batch = remaining_index[start_idx:end_idx]
    
        for i, rect in enumerate(rect_info):
            cluster_data = points[labels == i]
            cluster_points = cluster_data[['X (m)', 'Y (m)', 'depth (m)']].values
            updated = True
    
            while updated:
                # Check which points in the current batch belong to the current rectangle
                belongs_to_rect = np.array([check_point_in_rectangle(pt, rect, distance_limit) for pt in current_batch])
                # print("belongs_to_rect = ", belongs_to_rect)
                # Ensure belongs_to_rect is a NumPy array of boolean type
                belongs_to_rect = np.array(belongs_to_rect, dtype=bool)
                    
                # Select points belonging to the rectangle
                new_points = current_batch[belongs_to_rect]
                assigned_indices = index_batch[belongs_to_rect]  # Map back to global indices
                
                if len(new_points) > 0:
                    # Update the labels of the assigned points
                    # Ensure `labels` is explicitly a copy
                    labels = labels.copy()                    
                    # Use .loc to update the labels
                    labels.loc[assigned_indices] = i

                    # Update the cluster and rectangle
                    cluster_points, new_corners, rect = update_cluster_and_rectangle(cluster_points, new_points)
                
                    # Update remaining points
                    current_batch = current_batch[~belongs_to_rect]
                    index_batch = index_batch[~belongs_to_rect]

                    # Update rectangle info after processing each rectangle
                    rect_info[i] = rect
                    rectangles[i] = new_corners
                else:
                    updated = False
    
            # Break the outer loop if there are no remaining points in the current_batch
            if len(current_batch) == 0:
                break
            
    remaining_data = points[labels == -1]  # Unclustered points
    # print("After len(remaining_data) = ", len(remaining_data))
    
    return rectangles, rect_info, remaining_data, labels

def reorder_rectangle_points(rect):
    centroid = np.mean(rect, axis=0)
    angles = np.arctan2(rect[:, 1] - centroid[1], rect[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    return rect[sorted_indices]

def calculate_dip_strike_and_direction(rect):
    """
    Calculate dip, dip direction, and strike of a rectangle plane in Earth coordinates,
    following the right-hand rule.

    Parameters:
        rect (ndarray): 4x3 array representing the corners of a rectangle.

    Returns:
        tuple: Dip (degrees), Strike (degrees), Dip Direction (degrees)
    """
    # Step 1: Fit the plane using the first three points of the rectangle
    p1, p2, p3 = rect[:3]
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector

    # Step 2: Calculate Dip (angle between the normal vector and vertical axis)
    dip = np.degrees(np.arccos(abs(normal_vector[2])))  # Dip angle relative to the horizontal plane

    # Step 3: Calculate Dip Direction (azimuth of steepest descent)
    # Project the normal vector onto the horizontal plane
    horizontal_projection = np.array([normal_vector[0], normal_vector[1]])
    if np.linalg.norm(horizontal_projection) == 0:  # Handle case where plane is horizontal
        dip_direction = 0  # Arbitrary (no steepest descent for horizontal plane)
    else:
        dip_direction = np.degrees(np.arctan2(-horizontal_projection[0], -horizontal_projection[1]))
        if dip_direction < 0:
            dip_direction += 360

    # Step 4: Calculate Strike (90° counterclockwise from dip direction for right-hand rule)
    strike = (dip_direction - 90) % 360

    return dip, strike, dip_direction

def calculate_angle_between_vectors(vec1, vec2):
    """
    Calculate the angle (in degrees) between two vectors.
    
    Parameters:
        vec1 (ndarray): First vector.
        vec2 (ndarray): Second vector.
    
    Returns:
        float: Angle in degrees between the two vectors.
    """
    # Compute the dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes (norms) of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("One or both of the vectors have zero magnitude.")
    
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_vec1 * norm_vec2)
    
    # Ensure the value is within the valid range for arccos due to numerical precision issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # # Compute the angle in radians and convert to degrees
    # angle_radians = np.arccos(cos_theta)
    # angle_degrees = np.degrees(angle_radians)
    
    return cos_theta

def get_edge_points_along_strike(center, strike, rect):
    """
    Get the two points on the edges of the rectangle along the strike direction.

    Parameters:
        center (ndarray): 1x3 array representing the center of the rectangle.
        strike (float): Strike direction in degrees.
        rect (ndarray): 4x3 array representing the corners of the rectangle.

    Returns:
        tuple: Two 1x3 arrays representing the positions of the two points on the edges.
    """
    # Convert strike direction to a unit vector in 3D
    strike_rad = np.radians(strike)
    strike_vector = np.array([np.sin(strike_rad), np.cos(strike_rad), 0])  # Strike vector in 3D

    # Project the corners of the rectangle onto the strike line
    projections = []
    for corner in rect:
        # Vector from center to corner
        corner_vector = corner - center

        # Project the corner vector onto the strike direction
        projection_length = np.dot(corner_vector, strike_vector)
        projection_point = center + projection_length * strike_vector
        projections.append(projection_point)

    # Sort projections based on their distance along the strike direction
    projections = np.array(projections)
    distances = np.dot(projections - center, strike_vector)
    sorted_indices = np.argsort(distances)

    # Return the two farthest projections (one on each side of the center)
    return projections[sorted_indices[0]], projections[sorted_indices[-1]]

def save_rectangles_to_file(rectangles, rect_info, filename):
    with open(filename, 'w') as f:
        f.write(f"Number of faults: {len(rectangles)}\n\n")
        for i, (rect, info) in enumerate(zip(rectangles, rect_info)):
            dip, strike, dip_direction = calculate_dip_strike_and_direction(rect)
            
            intersection_points = get_edge_points_along_strike(info['center'], strike, rect)
            
            f.write(f"Rectangle #{i + 1}:\n")
            f.write(f"num_points: {info['num_points']}\n")            
            
            f.write(f"rect_1 = [{rect[0][0]:.6f}," 
                    f"{rect[0][1]:.6f}, {rect[0][2]:.6f}]\n")
            f.write(f"rect_2 = [{rect[1][0]:.6f}," 
                    f"{rect[1][1]:.6f}, {rect[1][2]:.6f}]\n")
            f.write(f"rect_3 = [{rect[2][0]:.6f}," 
                    f"{rect[2][1]:.6f}, {rect[2][2]:.6f}]\n")
            f.write(f"rect_4 = [{rect[3][0]:.6f}," 
                    f"{rect[3][1]:.6f}, {rect[3][2]:.6f}]\n")
                         
            f.write(f"center = [{info['center'][0]:.6f}, {info['center'][1]:.6f}, {info['center'][2]:.6f}]\n")
            f.write(f"point_1 = [{intersection_points[0][0]:.6f}," 
                    f"{intersection_points[0][1]:.6f}, {intersection_points[0][2]:.6f}]\n")
            f.write(f"point_2 = [{intersection_points[1][0]:.6f}," 
                    f"{intersection_points[1][1]:.6f}, {intersection_points[1][2]:.6f}]\n")            
            
            f.write(f"  Length: {info['length']:.6f}, Width: {info['width']:.6f}\n")
            f.write(f"  Dip: {dip:.2f}°, Strike: {strike:.2f}°, Dip Direction: {dip_direction:.2f}°\n\n")

def adjust_rgb_color(rgb_string, adjustment):
    """
    Adjust RGB color values by a specific factor.
    - rgb_string: The RGB string, e.g., 'rgb(68, 1, 84)'
    - adjustment: A tuple (dr, dg, db) to adjust (R, G, B) values
    Returns: A new adjusted RGB string.
    """
    # Extract RGB values using regex
    match = re.match(r'rgb\((\d+), (\d+), (\d+)\)', rgb_string)
    if not match:
        raise ValueError("Invalid RGB string format")

    # Convert extracted values to integers and adjust them
    r, g, b = map(int, match.groups())
    r = max(0, min(255, r + adjustment[0]))
    g = max(0, min(255, g + adjustment[1]))
    b = max(0, min(255, b + adjustment[2]))

    # Return the new RGB string
    return f'rgb({r}, {g}, {b})'

def plot_2d_rectangles_in_3d_plotly(points, rectangles, labels, well, filename):
    
    fig = go.Figure()
    colormap = 'viridis'
    # colormap = qualitative.Dark2
    num_rects = len(rectangles)
    colors = sample_colorscale(colormap, [i / num_rects for i in range(num_rects)])
    # print('colors = ', colors)
    
    # # # Adjust colors by adding (or subtracting) intensity
    # # adjustment = (30, -20, 10)  # Example adjustment for (R, G, B)
    # # colors = [adjust_rgb_color(color, adjustment) for color in colors]
    # # print("Adjusted colors:", colors)
    
    colors = ['rgb(68, 1, 84)', 'rgb(255, 255, 0)']  # Replace with your desired colors
    # Map labels to the corresponding colors
    label_to_color = {label: colors[label] for label in range(num_rects)}
    # Add a default color for the label -1
    label_to_color[-1] = 'rgb(255, 0, 0)'  # You can choose any default color for -1
    # print('label_to_color = ', label_to_color)

    # Ensure the number of colors matches the number of rectangles
    if len(colors) < num_rects:
        raise ValueError("Not enough colors provided for the number of rectangles.")

    # Generate marker colors by mapping labels
    marker_colors = [label_to_color[label] for label in labels]
    
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        
        marker=dict(
            size=4,
            color=marker_colors,  # Use numeric values for coloring
            colorscale=colormap,  # Color scale
            opacity=0.8,
            showscale=False  # Hide the colorbar
        ),
        # marker_line=dict(color='black', width=0.5),  # Outline the points with a black line
        name='Events'  # Trace name
    ))
    for i, rect in enumerate(rectangles):
        rect = np.vstack([rect, rect[0]])
        fig.add_trace(go.Scatter3d(
            x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
            mode='lines',
            line=dict(color=colors[i], width=4),
            name=f'Fitted Fault {i + 1}'
        ))
        
        # Add filled surface for the rectangle
        fig.add_trace(go.Mesh3d(
            x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
            color=colors[i], 
            opacity=0.5,
            # name=f'Fault Surface {i + 1}'
        ))
        
    well.plot_trajectory_plotly(fig)
    
    # Define custom tick values and labels
    x_tickvals = [3.348e5, 3.353e5, 3.358e5]  # Tick values for x-axis
    x_ticktext = ['3.348e5', '3.353e5', '3.358e5']  # Corresponding labels for x-axis
    
    y_tickvals = [4.26315e6, 4.2633e6, 4.26345e6]  # Tick values for y-axis
    y_ticktext = ['4.26315e6', '4.2633e6', '4.26345e6']  # Corresponding labels for y-axis
    
    z_tickvals = [0, 1000, 2000]  # Tick values for z-axis
    z_ticktext = ['0', '1000', '2000']  # Corresponding labels for z-axis
    
    # Update layout for axes and titles
    fig.update_layout(
        legend=dict(
            font=dict(
                family='Arial',  # Set the font family
                size=14,  # Set the font size for the trace name
                color='black'  # Set the font color for the trace name
            ),
            # title=dict(
            #     text='Legend Title',  # Set legend title (optional)
            #     font=dict(size=14, color='black')  # Set font for the legend title
            # ),
            traceorder='normal'  # Control the order of traces in the legend
        ),
        scene=dict(
            xaxis=dict(
                title='Easting (m)',
                tickvals=x_tickvals,  # Custom tick positions for x-axis
                ticktext=x_ticktext,  # Custom tick labels for x-axis
                title_font=dict(size=14, family='Arial', color='black'),
                tickfont=dict(size=12, family='Arial', color='black')
            ),
            yaxis=dict(
                title='Northing (m)',
                tickvals=y_tickvals,  # Custom tick positions for y-axis
                ticktext=y_ticktext,  # Custom tick labels for y-axis
                title_font=dict(size=14, family='Arial', color='black'),
                tickfont=dict(size=12, family='Arial', color='black')
            ),
            zaxis=dict(
                title='Depth (m)',
                tickvals=z_tickvals,  # Custom tick positions for z-axis
                ticktext=z_ticktext,  # Custom tick labels for z-axis
                title_font=dict(size=14, family='Arial', color='black'),
                tickfont=dict(size=12, family='Arial', color='black'),
                autorange='reversed'  # Reverse Z-axis for depth
            ),
            aspectratio=dict(x=2, y=2, z=1)  # Set custom aspect ratio
        ),
        font=dict(size=12, family='Arial', color='black'),  # Font for the entire figure (including titles)
        # width=1920, height=1080,
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
    )
    
    fig.write_html(filename)
    
    # # Show the plot in an interactive window
    # fig.show()
    
    # Optionally, open the plot in a browser
    absolute_filename = os.path.abspath(filename)
    import webbrowser
    webbrowser.open(f"file://{absolute_filename}")
    # webbrowser.open(filename)     

def plot_projected_points_and_rectangle(rectangles, projected_points, local_coordinates, times, labels, well, filename):
    """
    Plot the projected points and the fitted rectangle in 3D and 2D.
    Args:
        rect: Corners of the rectangle in 3D.
        projected_points: Projected 3D points.
        local_coordinates: 2D local coordinates of the projected points.
    """
    start_time = pd.to_datetime("2022-04-21 13:53:43")  # Specify your start time
    end_time = pd.to_datetime("2022-04-21 18:23:21")
    
    # Generate numeric timestamps for the color mapping
    # times_numeric = times.astype('int64') / 1e9  # Convert datetime to UNIX timestamp in seconds    
    times_numeric = times[(times >= start_time) & (times <= end_time)].astype('int64') / 1e9  # Convert datetime to UNIX timestamp in seconds

    
    # Choose evenly spaced ticks within the filtered range
    num_ticks = 5  # Specify the number of ticks you want
    # tick_indices = np.linspace(0, len(times_numeric) - 1, num_ticks, dtype=int)  # Evenly spaced indices
    
    # Generate evenly spaced values within the range of times_numeric    
    tick_values = np.linspace(times_numeric.min(), times_numeric.max(), num_ticks)
    # Find the closest indices in times_numeric for each tick value
    tick_indices = [np.abs(times_numeric - tick_value).argmin() for tick_value in tick_values]
    
    tickvals = times_numeric[tick_indices].tolist()  # Numeric tick values
    ticktext = times[tick_indices].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()  # Formatted datetime strings
    
    projected_points_plot = projected_points[labels != -1]
        
    fig = go.Figure()    
    fig.add_trace(go.Scatter3d(
        # x=projected_points[:, 0], y=projected_points[:, 1], z=projected_points[:, 2],
        x=projected_points_plot[:, 0], y=projected_points_plot[:, 1], z=projected_points_plot[:, 2],
        mode='markers',
        marker=dict(
            size=5,         
            color=times_numeric,  # Use the UNIX timestamp as the color
            colorscale='Viridis',  # Choose a perceptually uniform colormap
            opacity=0.8,
            colorbar=dict(
                title='Time',  # Title for the colorbar  
                tickvals=tickvals,  # Use only selected numeric values for ticks
                ticktext=ticktext,  # Corresponding formatted strings for tick labels              
                x=1.1,  # Shift the colorbar to the right (default is 1)
                y=0.5,  # Vertical position of the colorbar (0 is bottom, 1 is top)
                len=0.75,  # Length of the colorbar (fraction of the plot height)
                title_font=dict(size=12),  # Font size for the title
                tickfont=dict(size=10)  # Font size for the ticks
            )
        ),
        name='Projected Points'
    ))
    
    for i, rect in enumerate(rectangles):
        rect = np.vstack([rect, rect[0]])
        fig.add_trace(go.Scatter3d(
            x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'Fitted Fault {i + 1}'
        ))
        
        # Add filled surface for the rectangle
        fig.add_trace(go.Mesh3d(
            x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
            color='white', 
            opacity=0.5,
            # name=f'Fault Surface {i + 1}'
        ))   
            
    well.plot_trajectory_plotly(fig)
    
    # Update layout for axes and titles
    fig.update_layout(
        legend=dict(
            x=1.2,  # Move the legend further to the right
            y=0.5,  # Center the legend vertically
            xanchor='left',
            yanchor='middle',
            font=dict(
                family='Arial',  # Set the font family
                size=14,  # Set the font size for the trace name
                color='black'  # Set the font color for the trace name
            ),
            # title=dict(
            #     text='Legend Title',  # Set legend title (optional)
            #     font=dict(size=14, color='black')  # Set font for the legend title
            # ),
            traceorder='normal'  # Control the order of traces in the legend
        ),
        scene=dict(
            xaxis_title='Easting (m)',
            yaxis_title='Northing (m)',
            zaxis_title='Depth (m)',
            zaxis=dict(autorange='reversed'),
            aspectratio=dict(x=2, y=2, z=1) 
        ),
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
    )

    # Define custom tick values and labels
    x_tickvals = [3.348e5, 3.353e5, 3.358e5]  # Tick values for x-axis
    x_ticktext = ['3.348e5', '3.353e5', '3.358e5']  # Corresponding labels for x-axis
    
    y_tickvals = [4.26315e6, 4.2633e6, 4.26345e6]  # Tick values for y-axis
    y_ticktext = ['4.26315e6', '4.2633e6', '4.26345e6']  # Corresponding labels for y-axis
    
    z_tickvals = [0, 1000, 2000]  # Tick values for z-axis
    z_ticktext = ['0', '1000', '2000']  # Corresponding labels for z-axis

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Easting (m)',
                tickvals=x_tickvals,  # Custom tick positions for x-axis
                ticktext=x_ticktext,  # Custom tick labels for x-axis
                title_font=dict(size=14, family='Arial', color='black'),
                tickfont=dict(size=12, family='Arial', color='black')
            ),
            yaxis=dict(
                title='Northing (m)',
                tickvals=y_tickvals,  # Custom tick positions for y-axis
                ticktext=y_ticktext,  # Custom tick labels for y-axis
                title_font=dict(size=14, family='Arial', color='black'),
                tickfont=dict(size=12, family='Arial', color='black')
            ),
            zaxis=dict(
                title='Depth (m)',
                tickvals=z_tickvals,  # Custom tick positions for z-axis
                ticktext=z_ticktext,  # Custom tick labels for z-axis
                title_font=dict(size=14, family='Arial', color='black'),
                tickfont=dict(size=12, family='Arial', color='black'),
                autorange='reversed'  # Reverse Z-axis for depth
            ),
            aspectratio=dict(x=2, y=2, z=1)  # Set custom aspect ratio
        ),
        font=dict(size=12, family='Arial', color='black'),  # Font for the entire figure (including titles)
        # width=1200,
        # height=1000,
        margin=dict(l=0, r=0, t=0, b=0)  # Adjust margins
    )
    
    fig.write_html(filename)
    
    # # Show the plot in an interactive window
    # fig.show()
    
    # Optionally, open the plot in a browser
    absolute_filename = os.path.abspath(filename)
    import webbrowser
    webbrowser.open(f"file://{absolute_filename}")
    # webbrowser.open(filename)        

# Main Script
if __name__ == "__main__":
    # Initialize well object for 16A-32
    well_16a = Well(
        name="Well 16A-32",
        file_path='./well16A(78)-32.xlsx',
        start_row=1,
        columns=[0, 1, 2],
        column_names=['True Vertical Distance', 'Northing', 'Easting'],
        plot_color='blue'
    )
    
    # Read the Excel file
    file_path = "./well16A/FORGECatalog_April22_well16AStage 3.xlsx"
    data = pd.read_excel(file_path)
    
    # Ensure that the dateTime column is in datetime format
    data['dateTime'] = pd.to_datetime(data['dateTime'])
    times = data['dateTime']
    # Convert datetime values to UNIX timestamps (seconds since 1970-01-01): a numerical format for coloring
    data['dateTime_numeric'] = data['dateTime'].astype('int64')
    
    # Sort the data by the dateTime column in ascending order
    # data_sorted = data.sort_values(by='dateTime', ascending=True).reset_index(drop=True)
    
    # For example, if you want to restrict the time range:
    start_time = pd.to_datetime("2022-04-21 13:53:43")  # Specify your start time
    # end_time = pd.to_datetime("2022-04-21 17:16:37")
    end_time = pd.to_datetime("2022-04-21 15:23:21")
    
    # Assign labels based on the conditions
    data['label'] = -1  # Default label
    data.loc[(data['dateTime'] >= start_time) & (data['dateTime'] <= end_time), 'label'] = 0
    # data.loc[data['dateTime'] > end_time, 'label'] = -1
    
    batch_size = 10  # Define the batch size (e.g., 10)
    distance_limit = 15
    rectangles_1, rect_info_1, remaining_data, labels_1 = \
        iterative_clustering_and_update(data, batch_size, distance_limit)
    
    labels_1 = labels_1[labels_1 != -1].reset_index(drop=True)   
    
    ordered_rectangles_1 = [reorder_rectangle_points(rect) for rect in rectangles_1]

    remaining_points = remaining_data[['X (m)', 'Y (m)', 'depth (m)']].values
    
    eps = 50
    min_samples = 10
    rectangles_2, labels_2, rect_info_2 = \
        fit_2d_rectangles_to_3d_points(remaining_points, eps, min_samples)
        
    labels_2[labels_2 > -1] += 1
    labels_2 = pd.Series(labels_2)
    # rectangles, labels, rect_info = fit_2d_rectangles_to_3d_points(data) 
    
    ordered_rectangles_2 = [reorder_rectangle_points(rect) for rect in rectangles_2]
    
    ordered_rectangles = ordered_rectangles_1 + ordered_rectangles_2
    rect_info = rect_info_1 + rect_info_2
    
    # Concatenate (combine both)
    labels = pd.concat([labels_1, labels_2], ignore_index=True)
    
    data['label'] = labels
    
    base_filename = "./well16A/2022_well16AStage 3"
    
    points = data[['X (m)', 'Y (m)', 'depth (m)']].values
    # plot_2d_rectangles_in_3d_plotly(points, ordered_rectangles, labels, well_16a, filename=f"{base_filename}.html")

    projected_points, local_coordinates = project_points_onto_plane(points, labels, rect_info)
    
    plot_projected_points_and_rectangle(ordered_rectangles, projected_points, local_coordinates, times, labels, well_16a, filename=f"{base_filename}_projection.html")

    # Save fault information
    save_rectangles_to_file(ordered_rectangles, rect_info, filename=f"{base_filename}.txt")
