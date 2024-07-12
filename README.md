# mesh_landmarking
A tool for landmarking a 3D mesh written in Python.

https://github.com/user-attachments/assets/47768b1e-43e7-4677-80ce-36d77c0e704e

## Usage

`python mesh_landmarking.py <path to mesh> <path to landmarks> -nl <max number of landmarks (default inf)> <-b to store the landmarks as barycentric>`

example:

`python mesh_landmarking.py meshes/mesh.obj landmarks/bary.npy -b`

To save the result, quit the application by closing the window or pressing `q`.

## Notes

This tool uses Pyvista and PyvistaQt to interactively place, move, swap, remove, and resize landmarks on a 3D surface mesh. Landmarks (or markers) are a popular mean to annotate correspondences accross multiple meshes with different topology. Landmarks are extensively used in the creation of 3D Morphable Models for instance.

## Controls

The controls are displayed in the console when launching the application.

| Key    | Action |
| -------- | ------- |
| d  | Place a landmark under the cursor (if there is some surface)    |
| BackSpace | Remove the last placed landmark (in red)     |
| g    | Select ("grab") the landmark under the cursor (if there is one)   |
| b    | Flag the selected landmark as "bad"   |
| h    | Unselect the selected landmark (if one is selected)  |
| s    | Swap selected landmark with the landmark under the mouse cursor |
| c    | Display landmark index in the terminal |
| Numpad Plus    | Increase landmark size |
| Numpad Minus    | Decrease landmark size |
| Space    | Toggle mesh visibility |

## Color coding

Landmarks are color coded to ease the process.

| Color    | Meaning |
| -------- | ------- |
| <span style="color:lightblue">blue</span>  | Default |
| <span style="color:red">red</span>  | Last placed |
| <span style="color:green">green</span>  | Selected |
| <span style="color:black">black</span>  | Flagged as bad |


## Requirements
- python 3.7+
- pyvista
- pyvistaqt
- trimesh
- numpy
- click
