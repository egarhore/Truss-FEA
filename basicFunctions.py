import numpy as np
import matplotlib.pyplot as plt


# Function to create the geometry of the structure
def create_geometry(frame_length, frame_height, mesh_size, b_type):
    # Check the dimension of the length
    if np.remainder(frame_length, mesh_size) != 0:
        mesh_size = np.divide(frame_length, np.ceil(np.divide(frame_length, mesh_size)))
    # Ensure check dimension also matches height
    if np.remainder(frame_height, mesh_size) != 0:
        print('Check dimensions')
        return 1

    # Number of nodes in x and y dimensions and total nodes
    nx = int(frame_length/mesh_size + 1)
    ny = int(frame_height/mesh_size + 1)
    total_nodes = int(nx * ny)

    # Initialise the coordinates of every node to zero
    coords = np.zeros([total_nodes, 2])
    x_coord = 0
    node_count = 0

    # Assemble the number of nodes
    for i in range(nx):
        y_coord = 0
        for j in range(ny):
            coords[node_count, [0, 1]] = np.array([x_coord, y_coord])
            y_coord = y_coord + mesh_size
            node_count = node_count + 1
        x_coord = x_coord + mesh_size

    if b_type == 2 or b_type == 3:
        to_shift = np.arange(ny, total_nodes + 1, ny) - 1
        coords[to_shift, 0] = coords[to_shift, 0] + 0.001
    return coords, ny, nx


# Function t set the load and boundary condition
def set_load_bound(nx, ny, bt, lt, d_load):
    if bt == 1:
        # Fix the left top and bottom nodes
        bc_vector = np.zeros([nx * ny * 3, 2])
        bc_nodes = np.array([1, ny], dtype=int) - 1
        the_nodes = np.array([np.arange(1, 4, 1), np.arange((ny * 3) - 2, (ny * 3) + 1, 1)], dtype=int) - 1
        bc_vector[[the_nodes], 0] = 1
    elif bt == 2:
        # Fix the bottom left and set the bottom right to roller
        bc_vector = np.zeros([nx * ny * 3, 2])
        d_nodes = nx * ny
        d_nodes = d_nodes - ny + 1
        bc_nodes = np.array([0, d_nodes - 1], dtype=int)
        bc_vector[[0, 1, 2, int((d_nodes * 3) - 2), int((d_nodes * 3) - 1)], 0] = 1
    else:
        # Bottom left and right nodes fully fixed
        bc_vector = np.zeros([nx * ny * 3, 2])
        d_nodes = nx * ny
        d_nodes = d_nodes - ny + 1
        bc_nodes = np.array([0, d_nodes - 1], dtype=int)
        bc_vector[[0, 1, 2, int((d_nodes * 3) - 3), int((d_nodes * 3) - 2), int((d_nodes * 3) - 1)], 0] = 1

    # Set the load condition
    if lt == 1:
        # Load on the top right node
        l_vector = np.zeros([nx * ny * 3, 1])
        l_nodes = (nx * ny) - 1
        l_vector[(l_nodes * 3) + 1, 0] = d_load
    elif lt == 2:
        # Uniformly distributed load (top of the structure)
        l_vector = np.zeros([nx * ny * 3, 1])
        if bt == 1:
            d_nodes = np.arange(ny + ny, (nx * ny) + ny, ny) - 1
            l_nodes = np.arange(ny + ny, (nx * ny) + ny, ny) - 1
            d_nodes = (d_nodes * 3) + 1
            l_vector[d_nodes] = d_load/nx
        else:
            d_nodes = np.arange(ny, (nx * ny) + ny, ny) - 1
            l_nodes = np.arange(ny, (nx * ny) + ny, ny) - 1
            d_nodes = (d_nodes * 3) + 1
            l_vector[d_nodes] = d_load / nx
    elif lt == 3:
        # Point load at the bottom middle of the structure
        l_vector = np.zeros([nx * ny * 3, 1])
        bottom_nodes = np.arange(0, nx * ny, ny)
        is_odd = np.remainder(len(bottom_nodes), 2)
        if is_odd == 1:
            d_nodes = int(np.ceil(len(bottom_nodes) / 2))
            d_nodes = bottom_nodes[d_nodes - 1]
            l_nodes = d_nodes
            d_nodes = (d_nodes * 3) + 1
            l_vector[d_nodes] = -d_load
        else:
            d_nodes = int(np.ceil(len(bottom_nodes) / 2))
            d_nodes = np.array([d_nodes-1, d_nodes])
            d_nodes = bottom_nodes[d_nodes]
            l_nodes = d_nodes
            d_nodes = (d_nodes * 3) + 1
            l_vector[d_nodes] = -d_load / 2
    else:
        # Uniformly distributed load (bottom of the structure)
        l_vector = np.zeros([nx * ny * 3, 1])
        if bt == 1:
            d_nodes = np.arange(0 + ny, nx * ny, ny)
            l_nodes = np.arange(0 + ny, nx * ny, ny)
            d_nodes = (d_nodes * 3) + 1
            l_vector[d_nodes] = -d_load / nx
        else:
            d_nodes = np.arange(0 + ny, (nx * ny) - ny, ny)
            l_nodes = np.arange(0 + ny, (nx * ny) - ny, ny)
            d_nodes = (d_nodes * 3) + 1
            l_vector[d_nodes] = -d_load / nx
    return bc_nodes, bc_vector, l_nodes, l_vector


# Convert the user input to the design variable
def create_connectivity(nx, ny, connections, coordinates):
    # Create a matrix to create the connectivity matrix
    total_nodes = nx * ny
    # Connectivity matrix
    m = np.zeros([total_nodes, total_nodes], dtype=int)
    # Loop creating the connectivity matrix
    for i in range(len(connections)):
        x = connections[i, 0]
        y = connections[i, 1]
        if x < y:
            m[x, y] = 1
        elif x > y:
            m[y, x] = 1
        else:
            # Remove connection if node is connected to itself and warn the user
            print('Connection ' + str(i+1) + ' invalid')
            connections = np.delete(connections, obj=i, axis=0)

    # Get the total number of lines
    num_lines = len(connections)
    # Array to store the coordinates of each line format: [x1 y1 x2 y2]
    all_lines = np.zeros([num_lines, 4])
    # Array to store the nodes of each line format: [node1 node2]
    line_nodes = np.zeros([num_lines, 2], dtype=int)
    line_counter = 0

    # Loop over the connectivity matrix creating the lines (trusses)
    for i in range(total_nodes - 1):
        for j in range(total_nodes):
            if m[i, j] == 1:
                all_lines[line_counter, :] = np.array([coordinates[i, 0], coordinates[i, 1],
                                                       coordinates[j, 0], coordinates[j, 1]])
                line_nodes[line_counter, :] = np.array([i, j])
                line_counter = line_counter + 1

    # Convert to millimeters
    all_lines = all_lines / 1000
    # Return the values
    return m, all_lines, line_nodes


# Function for plotting the structure
def plot_connected_structures(coordinates, line_nodes, all_lines, disp_vector):
    # Get the total number of lines in the structure
    num_lines = len(all_lines)
    d_image, d_axis = plt.subplots()
    # Plot the undeformed structure
    for i in range(num_lines):
        x = np.array([all_lines[i, 0], all_lines[i, 2]])
        y = np.array([all_lines[i, 1], all_lines[i, 3]])
        if i == 0:
            d_axis.plot(x, y, color='green', linewidth=3, label='Undeformed')
        else:
            d_axis.plot(x, y, color='green', linewidth=3)
    # Plot the deformed structure if available
    if np.sum(disp_vector) != 0:
        # First remove the angular displacements
        to_remove = np.arange(2, len(disp_vector), 3)
        disp_vector = np.delete(disp_vector, obj=to_remove, axis=0)
        # Convert to a matrix to match the coordinate matrix
        rows = int(len(disp_vector) / 2)
        disp_vector = disp_vector.reshape(rows, 2)

        # Get a scale value to ensure deformation is visible
        sq = np.max(np.abs(disp_vector))
        sq = sq * 1000
        sq = np.ceil(sq)
        sq = 50 / sq
        power = -int(np.floor(np.log10(np.abs(sq))))
        factor = (10 ** power)
        sq = int(np.round(sq * factor) / factor)
        d_title = str(sq)
        sq = disp_vector * sq

        # Get the new coordinate
        new_coordinates = (coordinates / 1000) + sq
        # Get the deformed lines
        for i in range(num_lines):
            x = np.array([new_coordinates[line_nodes[i, 0], 0], new_coordinates[line_nodes[i, 1], 0]])
            y = np.array([new_coordinates[line_nodes[i, 0], 1], new_coordinates[line_nodes[i, 1], 1]])
            if i == 0:
                d_axis.plot(x, y, color='red', linewidth=3, label='Deformed')
            else:
                d_axis.plot(x, y, color='red', linewidth=3)

        # Add plot title
        d_title = 'Deformation scaled by: ' + d_title
        plt.title(d_title)
    # Display or save the image
    d_axis.legend()
    d_image.tight_layout()
    d_axis.axis('equal')
    d_axis.axis('off')

    plt.savefig('FEA Visualisation')
    # plt.show()


# Create a plot showing the nodes and the node number
def plot_nodes(coordinates):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    num_nodes = len(x)
    labels = np.arange(0, num_nodes)

    # plot the nodes
    d_image_0, d_axis_0 = plt.subplots()
    d_axis_0.scatter(x, y, alpha=0, edgecolors='black')

    for i in range(num_nodes):
        d_axis_0.text(x[i], y[i], str(labels[i]), bbox=dict(boxstyle='round', ec='black', fc='None'))

    # Display the graph
    plt.title('Graph showing the node numbers')
    d_image_0.tight_layout()
    d_axis_0.axis('equal')
    d_axis_0.axis('off')
    plt.savefig('Node visualisation')
    # plt.show()


# Function to reduce the connectivity matrix
def connectivity_reduction(a):
    # Declare a variable to store the index to be removed
    removed_index = np.zeros([len(a), 1], dtype=int)

    # Loop over the connectivity matrix checking for index to remove
    for i in range(len(a)):
        # Get row and column
        a_row = np.sum(a[i, :])
        a_column = np.sum(a[:, i])
        if a_row == 0 and a_column == 0:
            removed_index[i] = i

    # Remove the zeros values
    removed_index = np.array([i for i in removed_index if i != 0], dtype=int)
    reduced_a = np.delete(a, obj=removed_index, axis=0)
    reduced_a = np.delete(reduced_a, obj=removed_index, axis=1)

    return reduced_a, removed_index


# Convert removed_index to full scale
def reduced_to_full(r_i):
    full_index = np.array([], dtype=int)
    for i in range(len(r_i)):
        start = int(r_i[i] * 3)
        end = int((r_i[i] * 3) + 3)

        v = np.arange(start, end, 1)
        v = np.transpose(np.array(v, dtype=int))
        full_index = np.concatenate((full_index, v), axis=0)

    return full_index


# Calculate the length and angle of each line
def solve_line_length_angle(all_lines, geo_properties):
    num_lines = len(all_lines)
    line_length = np.zeros([num_lines, 1])
    line_angle = np.zeros([num_lines, 1])

    # Loop over the lines solving the length and angles
    for i in range(num_lines):
        x_len = all_lines[i, 2] - all_lines[i, 0]
        y_len = all_lines[i, 3] - all_lines[i, 1]
        diff_x = x_len
        diff_y = y_len
        x_len = diff_x * diff_x
        y_len = diff_y * diff_y
        line_length[i] = np.sqrt(x_len + y_len)
        if diff_x == 0:
            line_angle[i] = np.pi / 2
        else:
            line_angle[i] = np.arctan(diff_y / diff_x)

    # Calculate the volume of each line (truss)
    t = geo_properties['thickness']
    r = geo_properties['radius']
    p = geo_properties['density']
    line_mass = line_length
    line_hollows = np.pi * t * t * line_mass
    line_mass = np.pi * r * r * line_mass
    line_mass = line_mass - line_hollows
    line_mass = p * line_mass

    return line_length, line_angle, line_mass


def find_position(n, line_nodes):
    node_row = []
    node_col = []

    # iterate over the rows and columns
    for i in range(len(line_nodes)):
        for j in range(len(line_nodes[i])):
            if line_nodes[i, j] == n:
                node_row.append(i)
                node_col.append(j)

    return node_row, node_col
