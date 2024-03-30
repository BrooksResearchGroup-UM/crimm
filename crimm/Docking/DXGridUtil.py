import numpy as np
from scipy.io import FortranFile

DX_TEMPLATE = '''#Generated dx file for fft grid
object 1 class gridpositions counts {xd} {yd} {zd}
origin {min_x:e} {min_y:e} {min_z:e}
delta {spacing:e} 0.000000e+000 0.000000e+000
delta 0.000000e+000 {spacing:e} 0.000000e+000
delta 0.000000e+000 0.000000e+000 {spacing:e}
object 2 class gridconnections counts {xd} {yd} {zd}
object 3 class array type double rank 0 items {size} data follows
{values_str}
attribute "dep" string "positions"
object "regular positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3'''

def fill_dx(grid_vals, min_coords, spacing):
    values_str = generate_formatted_value_str(grid_vals)

    xd, yd, zd = grid_vals.shape
    min_x, min_y, min_z = min_coords
    template = DX_TEMPLATE.format(
        xd=xd, yd=yd, zd=zd, min_x=min_x, min_y=min_y, min_z=min_z,
        spacing=spacing, size=grid_vals.size, values_str=values_str
    )
    return template

def read_charmm_title_block(file_handle):
    l_width = 80  # CHARMM Default screen width
    # CHARMM title uses int32 for the two leading numbers
    int_size = 4  # size in byte
    title_b0 = file_handle.read(int_size)  # num characters in byte
    title_b4 = file_handle.read(int_size)  # num title lines in byte
    n_chars = int.from_bytes(title_b0, byteorder='little', signed=True)
    n_title_lines = int.from_bytes(title_b4, byteorder='little', signed=True)
    assert n_chars % int_size == 0
    n_chars -= int_size  # the last 4 bytes should be number of characters again
    # Read the characters from the title block and decode
    chars = file_handle.read(n_chars).decode('utf-8')
    title = tuple(
        chars[i:i+l_width].rstrip() for i in range(0, n_chars, l_width)
    )
    end_b = file_handle.read(int_size)  # num characters in byte
    assert end_b == title_b0
    assert n_title_lines == len(title)
    return title

def read_charmm_grid_bin(file_path):
    with open(file_path, 'rb') as fh:
        title_block = read_charmm_title_block(fh)
        with FortranFile(fh, 'r') as ff:
            grid_shape = ff.read_ints()  # NGrid, NGridX, NGridY, NGridZ
            center = ff.read_reals()  # (x, y, z)
            XGridMax, YGridMax, ZGridMax, DGrid, GridForce = ff.read_reals()
            vdw_radii = ff.read_reals()
            potential = ff.read_reals()
            potential = potential.reshape(grid_shape, order="F")

    data_dict = {
        'title': title_block,
        'center': center,
        'max_dim':  np.array([XGridMax, YGridMax, ZGridMax]),
        'd_grid': DGrid,
        'grid_force': GridForce,
        'vdw_radii': vdw_radii,
        'grid_vals': potential
    }
    return data_dict

# def generate_formatted_value_str(grid, values_per_line = 6):
#     remainder = grid.size % values_per_line
#     if remainder:
#         reshaped = grid[:-remainder-1].reshape(-1,values_per_line)
#         extra_line = grid[-remainder:]
#     else:
#         reshaped = grid.reshape(-1,values_per_line)
#         extra_line = None

#     val_str = ''
#     for arr_line in reshaped:
#         val_str += ' '.join(f'{val:e}' for val in arr_line)+'\n'
#     if extra_line:
#         val_str += ' '.join(f'{val:e}' for val in extra_line)+'\n'
#     return val_str

def generate_formatted_value_str(grid, values_per_line = 6):
    val_str = ''
    for i, val in enumerate(grid.flatten(), start = 1):
        val_str += f'{val:e} '
        if i%values_per_line == 0:
            val_str += '\n'
    return val_str

def save_grid_dict_to_dx(grid_dict, prefix='grid', elec_only=False):
    last_three_grids = ['H-donor','H-acceptor','elec']
    min_coords = grid_dict['center'] - grid_dict['max_dim']/ 2
    spacing = grid_dict['d_grid']
    n_remainder_grids = len(grid_dict['vdw_radii']) - len(grid_dict['grid_vals'])
    file_names = []
    if elec_only:
        grids = grid_dict['grid_vals'][-1:]
    else:
        grids = grid_dict['grid_vals']
        # 26 vdw grids
        for vdw_r in grid_dict['vdw_radii']:
            file_names.append(f'{prefix}-vdw-{vdw_r}.dx')

    # electrostatic grid is last
    file_names.append(f'{prefix}-elec.dx')

    for filename, grid_val in zip(file_names, grids):
        dx_str = fill_dx(grid_val, min_coords, spacing)
        with open(filename, 'w') as fh:
            fh.write(dx_str)