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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.colors import qualitative
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
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
        
# Function Definitions
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
            "num_points": len(cluster_points)
        })

    return rectangles, labels, rect_info

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
            
# def save_rectangles_to_file(rectangles, rect_info, filename):
#     with open(filename, 'w') as f:
#         f.write(f"Number of faults: {len(rectangles)}\n\n")
#         for i, (rect, info) in enumerate(zip(rectangles, rect_info)):
#             dip, strike, dip_direction = calculate_dip_strike_and_direction(rect)
#             f.write(f"Rectangle #{i + 1}:\n")
#             f.write(f"num_points: {info['num_points']}\n")
#             # f.write(f"  Center: {info['center']}\n")
#             f.write(f"  Center: {info['center'][0]:.6f}, {info['center'][1]:.6f}, {info['center'][2]:.6f}\n")
#             f.write(f"  Length: {info['length']:.6f}, Width: {info['width']:.6f}\n")
#             f.write(f"  Dip: {dip:.2f}°, Strike: {strike:.2f}°, Dip Direction: {dip_direction:.2f}°\n\n")

def plot_2d_rectangles_in_3d_plotly(points, rectangles, labels, well, filename):
    
    fig = go.Figure()
    colormap = 'viridis'
    # colormap = qualitative.Dark2
    num_rects = len(rectangles)
    colors = sample_colorscale(colormap, [i / num_rects for i in range(num_rects)])
    
    # # # Adjust colors by adding (or subtracting) intensity
    # # adjustment = (30, -20, 10)  # Example adjustment for (R, G, B)
    # # colors = [adjust_rgb_color(color, adjustment) for color in colors]
    # # print("Adjusted colors:", colors)
    
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
    # print('marker_colors = ', marker_colors)
    
    # # Define unique RGB colors and map them to numeric values
    # rgb_to_numeric = {
    #     'rgb(255, 0, 0)': 0,  # Red for -1 or other special labels
    #     'rgb(68, 1, 84)': 1,   # Example for another color
    #     # Add more RGB color mappings if you have more unique colors
    # }    
    # # Convert the RGB colors in marker_colors to their corresponding numeric values
    # numeric_marker_colors = [rgb_to_numeric[color] for color in marker_colors]
    
    # # Map labels to the corresponding colors as numeric values
    # label_to_color = {label: idx for idx, label in enumerate(range(num_rects))}
    # # Add a default color for the label -1 (use a distinct numeric value for -1)
    # label_to_color[-1] = -1  # Assign -1 to indicate this label needs a specific color
    # # Create the marker_colors using numeric labels
    # marker_colors = [label_to_color[label] for label in labels]
    
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',        
        marker=dict(
            size=4,
            color=marker_colors,  # Use numeric values for coloring
            colorscale=colormap,  # Color scale
            opacity=0.8,
            showscale=False,  # Hide the colorbar
            colorbar=dict(
                title_font=dict(size=14, family='Arial', color='black'),  # Set font for the colorbar title
                tickfont=dict(size=14, family='Arial', color='black'),  # Font for tick labels in colorbar
                # thickness=15,  # Set the width of the colorbar              
                x=1.1,  # Shift the colorbar to the right (default is 1)
                y=0.5,  # Vertical position of the colorbar (0 is bottom, 1 is top)
                len=0.75  # Length of the colorbar (fraction of the plot height)
            )
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
        
    # for i, rect in enumerate(rectangles):
    #     if i==0:
    #         rect = np.vstack([rect, rect[0]])
    #         fig.add_trace(go.Scatter3d(
    #             x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
    #             mode='lines',
    #             line=dict(color=colors[i], width=4),
    #             name=f'Fitted Fault #{i + 1}'
    #         ))
            
    #         # Add filled surface for the rectangle
    #         fig.add_trace(go.Mesh3d(
    #             x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
    #             color=colors[i],
    #             opacity=0.5,
    #             # name=f'Fault Surface {i + 1}'
    #         ))
        
    # for i, rect in enumerate(rectangles):
    #     if i==2:            
    #         rect = np.vstack([rect, rect[0]])
    #         fig.add_trace(go.Scatter3d(
    #             x=[rect[0, 0]],  # Wrap in a list
    #             y=[rect[0, 1]],  # Wrap in a list
    #             z=[rect[0, 2]],  # Wrap in a list
    #             mode='markers',
    #             marker=dict(size=5, 
    #                         color=1, 
    #                         colorscale=qualitative.Dark2, opacity=1.),
    #             name='Vertex 1'
    #         ))
    #         fig.add_trace(go.Scatter3d(
    #             x=[rect[1, 0]],  # Wrap in a list
    #             y=[rect[1, 1]],  # Wrap in a list
    #             z=[rect[1, 2]],  # Wrap in a list
    #             mode='markers',
    #             marker=dict(size=5, 
    #                         color=2, 
    #                         colorscale=qualitative.Dark2, opacity=1.),
    #             name='Vertex 2'
    #         ))
    #         # fig.add_trace(go.Scatter3d(
    #         #     x=[rect[2, 0]],  # Wrap in a list
    #         #     y=[rect[2, 1]],  # Wrap in a list
    #         #     z=[rect[2, 2]],  # Wrap in a list
    #         #     mode='markers',
    #         #     marker=dict(size=5, 
    #         #                 color=3, 
    #         #                 colorscale=qualitative.Dark2, opacity=1.),
    #         #     name='Vertex 3'
    #         # ))
    #         # fig.add_trace(go.Scatter3d(
    #         #     x=[rect[3, 0]],  # Wrap in a list
    #         #     y=[rect[3, 1]],  # Wrap in a list
    #         #     z=[rect[3, 2]],  # Wrap in a list
    #         #     mode='markers',
    #         #     marker=dict(size=5, 
    #         #                 color=4, 
    #         #                 colorscale=qualitative.Dark2, opacity=1.),
    #         #     name='Vertex 4'
    #         # ))
    #         fig.add_trace(go.Scatter3d(
    #             x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
    #             mode='lines',
    #             line=dict(color=colors[i+1], width=4),
    #             name=f'Fitted Fault {i + 1}'
    #         ))
            
    #         # Add filled surface for the rectangle
    #         fig.add_trace(go.Mesh3d(
    #             x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
    #             color=colors[i+1],
    #             opacity=0.5,
    #             # name=f'Fault Surface {i + 1}'
    #         ))
        
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

def plot_2d_rectangles_in_3d_matplotlib(points, rectangles, labels, well, filename):
    """
    Plot 2D rectangles and scatter points in a 3D space with consistent coloring.
    
    Parameters:
        points (ndarray): Nx3 array of point coordinates.
        rectangles (list): List of rectangle vertices as 4x3 arrays.
        labels (ndarray): Array of cluster labels corresponding to points.
        well: Well object with a plot_trajectory_matplotlib method.
        filename (str): Path to save the output plot.
    """
    # Create a larger figure to ensure axis labels are visible
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d') 

    # Normalize labels for colormap
    norm = Normalize(vmin=min(labels) + 1, vmax=max(labels) + 1)
    colormap = plt.cm.Dark2

    # Plot scatter with color labels
    scatter = ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2], 
        c=labels + 1,  # Adjust labels to start from 0
        cmap=colormap, 
        s=10, 
        alpha=0.8
    )
    
    # Create colorbar and set the ticks
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Fault Labels")
    cbar.set_ticks(np.arange(min(labels) + 1, max(labels) + 2))  # Ensure correct ticks

    # Plot rectangles with colors corresponding to labels
    for i, rect in enumerate(rectangles):
        # Close the rectangle by connecting the last point to the first
        rect = np.vstack([rect, rect[0]])
        fault_ID = i + 1
        color = colormap(norm(fault_ID))  # Get the corresponding color
        
        # Fill the rectangle with the corresponding color
        ax.plot_trisurf(
            rect[:, 0], rect[:, 1], rect[:, 2],
            triangles=[[0, 1, 2], [0, 2, 3]], color=color, alpha=1.
        )

    # Plot well trajectory
    well.plot_trajectory_matplotlib(ax)

    # Reverse the Z-axis to put the smaller values on top
    ax.set_zlim(np.max(points[:, 2]), np.min(points[:, 2]))

    ax.set_title("3D Microseismic Events and Fitting Faults with Well Trajectory")

    # Label the axes, making sure they are properly positioned within the figure
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Use tight_layout to automatically adjust the layout and make sure labels are visible
    plt.tight_layout()

    # # Save and show the plot
    plt.savefig(filename, dpi=300)
    # plt.show()
        
    # Save multiple rotation views
    rotation_angles = [(30, 30)]  # List of (elevation, azimuth)
    for i, (elev, azim) in enumerate(rotation_angles):
        ax.view_init(elev=elev, azim=azim)  # Set the view
        output_filename = f"{filename}_view{i+1}.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Saved plot with view (elev={elev}, azim={azim}) as {output_filename}")
    
    # Show the last plot (optional)
    plt.show()

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
    
    # Build the well dictionary
    well_dict = well_16a.to_dict()
    
    # # Access properties from the dictionary
    # depth_16a = well_dict["depth_16a"]
    # northing_16a = well_dict["northing_16a"]
    # easting_16a = well_dict["easting_16a"]
    # horizontal_16a = well_dict["horizontal_16a"]
    
    # Load microseismic data
    folder_path = "./well16A"
    file_paths = glob.glob(os.path.join(folder_path, "FORGECatalog_April24_well16A*.xlsx"))
    # Initialize an empty list to store all points
    all_points = []

    # Process each dataset
    for file in file_paths:
        # Extract the base file name
        full_file_name = os.path.basename(file).split(".")[0]

        # Find the part of the file name starting from "April24"
        start_index = full_file_name.find("April24")
        if start_index != -1:
            file_name = full_file_name[start_index:]  # Extract the substring
        else:
            file_name = full_file_name  # Fallback to full name if "April24" not found

        # print(f"Processing {file_name}")

        # Read the dataset
        data = pd.read_excel(file, usecols=['X (m)', 'Y (m)', 'depth (m)'])
        points = data.values  # Convert DataFrame to NumPy array

        # Append the points from this file to the all_points list
        all_points.append(points)

    # Combine all the points into a single NumPy array
    all_points = np.vstack(all_points)

    # Fit 2D rectangles in 3D space using the combined points
    eps = 50
    min_samples = 10
    # Fit rectangles
    rectangles, labels, rect_info = fit_2d_rectangles_to_3d_points(all_points, eps, min_samples)
    
    ordered_rectangles = [reorder_rectangle_points(rect) for rect in rectangles]

    # Generate unique filenames for each file (using a base name for consistency)
    base_filename = "./well16A/faultPlanes_2024_well16A"
    
    # Plot results
    plot_2d_rectangles_in_3d_plotly(all_points, ordered_rectangles, labels, well_16a, filename=f"{base_filename}.html")
    # plot_2d_rectangles_in_3d_matplotlib(all_points, ordered_rectangles, labels, well_16a, filename=f"{base_filename}.png")

    # Save fault information
    save_rectangles_to_file(ordered_rectangles, rect_info, filename=f"{base_filename}.txt")
