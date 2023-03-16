import numpy as np
from numpy.linalg import matrix_power
from basicFunctions import connectivity_reduction


# Check for intersections
def check_intersection(all_lines):
    # Set the fail value to false (meaning there is no intersection)
    all_lines = all_lines * 1000
    to_fail = False
    # Get the total number of lines
    num_lines = len(all_lines)
    for i in range(num_lines-1):
        # Get the coordinates
        x1 = all_lines[i, 0]
        x2 = all_lines[i, 2]
        y1 = all_lines[i, 1]
        y2 = all_lines[i, 3]
        a1 = y2 - y1
        b1 = x2 - x1
        for j in range(i + 1, num_lines):
            # Get the coordinates
            x3 = all_lines[j, 0]
            x4 = all_lines[j, 2]
            y3 = all_lines[j, 1]
            y4 = all_lines[j, 3]
            a2 = y4 - y3
            b2 = x4 - x3

            # Calculate for intersection
            a3 = y1 - y3
            b3 = x1 - x3
            a4 = y4 - y3
            b4 = x4 - x3

            det = (a4 * b1) - (b4 * a1)
            # det = (a1 * b2) - (a2 * b1)
            if det == 0:
                # Lines are parallel therefore check normal way
                # Compare with line 1
                # print(all_lines[i, :])
                # print(all_lines[j, :])
                # print('#####################################################################')
                if x1 == x3 and y1 == y3:
                    if x2 == x4 and min(y1, y2) < y4 < max(y1, y2):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif x2 == x4 and min(y3, y4) < y2 < max(y3, y4):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x1, x2) < x3 < max(x1, x2) and y2 == y4:
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x3, x4) < x2 < max(x3, x4) and y2 == y4:
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x1, x2) < x4 < max(x1, x2) and min(y1, y2) < y4 < max(y1, y2):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x3, x4) < x2 < max(x3, x4) and min(y3, y4) < y2 < max(y3, y4):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    else:
                        to_fail = False
                elif x2 == x3 and y2 == y3:
                    if x1 == x4 and min(y1, y2) < y4 < max(y1, y2):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif x1 == x4 and min(y3, y4) < y1 < max(y3, y4):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x1, x2) < x3 < max(x1, x2) and y1 == y4:
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x3, x4) < x1 < max(x3, x4) and y1 == y4:
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x1, x2) < x4 < max(x1, x2) and min(y1, y2) < y4 < max(y1, y2):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    elif min(x3, x4) < x1 < max(x3, x4) and min(y3, y4) < y1 < max(y3, y4):
                        to_fail = True
                        print('Intersecting trusses detected-A')
                        return to_fail
                    else:
                        to_fail = False
                # elif min(x1, x2) <= x3 <= max(x1, x2):
                #     if min(y1, y2) <= y3 <= max(y1, y2):
                #         # Make sure it is not node to node
                #         if x1 == x3 and y1 == y3:
                #             to_fail = 0
                #         elif x2 == x3 and y2 == y3:
                #             to_fail = 0
                #         elif x1 == x3 and y1 != y3:
                #             to_fail = 0
                #         elif x1 != x3 and y1 == y3:
                #             to_fail = 0
                #         elif x2 == x3 and y2 != y3:
                #             to_fail = 0
                #         elif x2 != x3 and y2 == y3:
                #             to_fail = 0
                #         else:
                #             print('Intersecting trusses detected-A')
                #             to_fail = 1
                #             return to_fail
                #     if min(x1, x2) <= x4 <= max(x1, x2):
                #         if min(y1, y2) <= y4 <= max(y1, y2):
                #             # Make sure it is not node to node
                #             if x1 == x4 and y1 == y4:
                #                 to_fail = 0
                #             elif x2 == x4 and y2 == y4:
                #                 to_fail = 0
                #             elif x1 == x4 and y1 != y4:
                #                 to_fail = 0
                #             elif x1 != x4 and y1 == y4:
                #                 to_fail = 0
                #             elif x2 == x4 and y2 != y4:
                #                 to_fail = 0
                #             elif x2 != x4 and y2 == y4:
                #                 to_fail = 0
                #             else:
                #                 print('Intersecting trusses detected-B')
                #                 to_fail = 1
                #                 return to_fail
                # # Compare with line 2
                # elif min(x3, x4) <= x1 <= max(x3, x4):
                #     if min(y3, y4) <= y1 <= max(y3, y4):
                #         # Make sure it is not node to node
                #         if x1 == x3 and y1 == y3:
                #             to_fail = 0
                #         elif x1 == x4 and y1 == y4:
                #             to_fail = 0
                #         elif x1 == x3 and y1 != y3:
                #             to_fail = 0
                #         elif x1 != x3 and y1 == y3:
                #             to_fail = 0
                #         elif x1 == x4 and y1 != y4:
                #             to_fail = 0
                #         elif x1 != x4 and y1 == y4:
                #             to_fail = 0
                #         else:
                #             print('Intersecting trusses detected-C')
                #             return to_fail
                #     if min(x3, x4) <= x2 <= max(x3, x4):
                #         if min(y3, y4) <= y2 <= max(y3, y4):
                #             # Make sure it is not node to node
                #             if x3 == x2 and y3 == y2:
                #                 to_fail = 0
                #             elif x4 == x2 and y4 == y2:
                #                 to_fail = 0
                #             elif x3 == x2 and y3 != y2:
                #                 to_fail = 0
                #             elif x3 != x2 and y3 == y2:
                #                 to_fail = 0
                #             elif x4 == x2 and y4 != y2:
                #                 to_fail = 0
                #             elif x4 != x2 and y4 == y2:
                #                 to_fail = 0
                #             else:
                #                 print('Intersecting trusses detected-D')
                #                 to_fail = 1
                #                 return to_fail
            else:
                ua = ((b2 * a3) - (a2 * b3)) / det
                ub = ((b1 * a3) - (a1 * b3)) / det

                x = check_intersect(ua, ub, x1, y1, x2, y2)
                if type(x) == int:
                    to_fail = False
                else:
                    y = x[1]
                    x = x[0]
                    if min(x1, x2) <= x <= max(x1, x2):
                        if min(x3, x4) <= x <= max(x3, x4):
                            if min(y1, y2) <= y <= max(y1, y2):
                                if min(y3, y4) <= y <= max(y3, y4):
                                    # Make sure it is not node to node
                                    if x1 == x and y1 == y and x3 == x and y3 == y:
                                        to_fail = False
                                    elif x1 == x and y1 == y and x4 == x and y4 == y:
                                        to_fail = False
                                    elif x2 == x and y2 == y and x3 == x and y3 == y:
                                        to_fail = False
                                    elif x2 == x and y2 == y and x4 == x and y4 == y:
                                        to_fail = False
                                    else:
                                        print('Intersecting trusses detected-E')
                                        to_fail = True
                                        return to_fail
    return to_fail


# To check for intersect
def check_intersect(a, b, x, y, m, n):
    if a < 0 or a > 1:
        return 0
    if b < 0 or b > 1:
        return 0

    x = x + a * (m - x)
    y = y + a * (n - y)

    return np.array([x, y])


# Function to ensure the structure is fully connected
def check_connection(con_matrix, force_node, bound_node):
    # Set the response to false
    to_fail = False

    # Convert connectivity matrix to full matrix
    con_matrix = con_matrix + np.transpose(con_matrix)

    # Reduce the connectivity matrix, remove the zeros
    reduced_a, removed_index = connectivity_reduction(con_matrix)

    # Perform the connectedness check
    con_sum = np.zeros_like(reduced_a)
    for i in range(len(reduced_a) - 1):
        con_sum = con_sum + matrix_power(reduced_a, i)

    # Fail the process if there is a zero is the connectedness matrix
    checker = np.isin(con_sum, 0)
    checker = True in checker
    if checker:
        print('The structure is not fully connected')
        to_fail = True
        return to_fail

    # Check if the force nodes are connected
    checker = np.isin(removed_index, force_node)
    checker = True in checker
    if checker:
        print('A loaded node is not connected')
        to_fail = True
        return to_fail

    # Check if the boundary nodes are connected
    checker = np.isin(removed_index, bound_node)
    checker = True in checker
    if checker:
        print('A boundary node is not connected')
        to_fail = True
        return to_fail
    return to_fail
