import datetime
import time

import numpy as np
import numpy.linalg as lin_np
from constraintEquations import check_intersection, check_connection
from basicFunctions import create_geometry, set_load_bound, create_connectivity, plot_connected_structures
from basicFunctions import solve_line_length_angle, connectivity_reduction, reduced_to_full, plot_nodes
from finite_element_analysis import assemble_stiffness_matrix, apply_penalty, redistribute_vector
from finite_element_analysis import solve_for_stiffness

# Start the timer
start_time = time.time()

d_properties = {}
with open('Input_Properties.txt') as f:
    for line in f:
        (key, val) = line.strip().split()
        d_properties[key] = val

# Get size of the structure
totalLength = float(d_properties['Total_Length(mm)'])
totalHeight = float(d_properties['Total_Height(mm)'])
meshSize = float(d_properties['Mesh_Size(mm)'])

# Get the material properties
E = float(d_properties['Young_Modulus(Pa)'])
r = float(d_properties['Radius(mm)']) / 1000
t = float(d_properties['Thickness(mm)']) / 1000
p = float(d_properties['Density(kg/m^3)']) / 1000

# Created dictionary to save the material properties (very unnecessary)
geoProperties = {'youngModulus': E, 'radius': r, 'thickness': t, 'density': p}

# Get the load and boundary condition
bType = int(d_properties['BoundaryType'])
lType = int(d_properties['LoadType'])
theLoad = float(d_properties['Load(N)'])

# Create the geometry of the structure
coords, nY, nX = create_geometry(totalLength, totalHeight, meshSize, bType)

# Get the total number of nodes and variables
nodeNumber = nY * nX

# Get the load and boundary condition
bcNodes, bcVector, lNodes, lVector = set_load_bound(nX, nY, bType, lType, theLoad)

# Obtain input and connectivity matrix and lines
plot_nodes(coords)
conns = np.loadtxt('Input_connections.txt', dtype=int)

print('----------------------------------------------------------------------------------------')
print('Solving...')
print('----------------------------------------------------------------------------------------')

conns_input = conns
conns, all_lines, line_nodes = create_connectivity(nX, nY, conns, coords)

# Ensure the design meets the requirement for analysis
if check_intersection(all_lines):
    exit()
if check_connection(conns, lNodes, bcNodes):
    exit()

# Solve for the length of all lines, angles of intersecting trusses and their masses
lineLength, lineAngle, lineMass = solve_line_length_angle(all_lines, geoProperties)
# Get the total mass of the structure (not including joints)
totalMass = np.sum(lineMass)

# Assemble stiffness matrix
nodeNumber_t = nodeNumber * 3
stiff_matrix, line_stiff_matrix, line_transform_matrix = \
    assemble_stiffness_matrix(all_lines, line_nodes, nodeNumber_t, geoProperties)

# Reduce the stiffness matricx, load vector and boundary matrix
reduced_conns, removed_index = connectivity_reduction(conns)
full_index = reduced_to_full(removed_index)
reduced_stiffness = np.delete(stiff_matrix, obj=full_index, axis=0)
reduced_stiffness = np.delete(reduced_stiffness, obj=full_index, axis=1)
reduced_lVector = np.delete(lVector, obj=full_index, axis=0)
reduced_bcVector = np.delete(bcVector, obj=full_index, axis=0)

# Apply penalty
num_nodes = len(reduced_lVector)
reduced_stiffness, reduced_lVector = apply_penalty(reduced_stiffness, reduced_lVector, reduced_bcVector, num_nodes)

# Solve for the displacement vector
reduced_dispVector = np.matmul(lin_np.inv(reduced_stiffness), reduced_lVector)

# Redistribute the displacement vector (set it to match the original size)
dispVector = np.zeros([nodeNumber_t, 1])
dispVector = redistribute_vector(dispVector, reduced_dispVector, removed_index, nodeNumber)

# Get the displacement in the x, y and theta direction
U1 = dispVector[np.array(np.arange(0, nodeNumber_t, 3), dtype=int)]
U2 = dispVector[np.array(np.arange(1, nodeNumber_t, 3), dtype=int)]
U3 = dispVector[np.array(np.arange(2, nodeNumber_t, 3), dtype=int)]

# Get the forces in the x, y and theta direction
F1 = lVector[np.array(np.arange(0, nodeNumber_t, 3), dtype=int)]
F2 = lVector[np.array(np.arange(1, nodeNumber_t, 3), dtype=int)]

# Calculate and display the specific stiffness of the structure
specific_stiffness = solve_for_stiffness(U1, U2, lNodes, F1, F2, totalMass)
specific_stiffness = '{:.4e}'.format(specific_stiffness)
print('Specific stiffness = ' + str(specific_stiffness) + ' N/kg')
print('----------------------------------------------------------------------')
# Complete process and show runtime
end_time = time.time()
run_time = end_time - start_time
run_time = str(datetime.timedelta(seconds=int(run_time)))
print('Process completed\n' + 'Total runtime = ' + run_time)
print('##############################################################################')

# Plot the structure
plot_connected_structures(coords, line_nodes, all_lines, dispVector)
