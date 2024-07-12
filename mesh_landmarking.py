import pyvista as pv
import pyvistaqt as pvqt
import trimesh as tm
import os
import numpy as np
import click
import sys

DISTANCE_THRESHOLD = 0.001
LAST_COLOR, SELECTED_COLOR, BASE_COLOR, BAD_COLOR = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)
LANDMARKS_SIZE = 0.0075
SCALING_FACTOR = 1.2

COMMANDS = dict(
    place_landmark=('d', 'key "d"', 'Place a landmark under the cursor (if there is some surface)'),
    remove_landmark=('BackSpace', 'BackSpace', 'Remove the last placed landmark (in red)'),
    select_landmark=('g', 'key "g"','Select the landmark under the cursor (if there is one). If the landmark is already selected, move it to under cursor.'),
    bad_landmark=('b', 'key "b"', 'Set the selected landmark as "bad", not usable later'),
    unselect_landmark=('h', 'key "h"', 'Unselect the selected landmark (if one is selected)'),
    swap_landmarks =('s', 'key "s"', 'Swap selected landmark position with the landmark under the mouse cursor'),
    check_landmark =('c', 'key "c"', 'Display landmark index in the terminal'),
    increase_landmark_size =('plus', 'Numpad Plus', 'Increase landmark size'),
    decrease_landmark_size =('minus', 'Numpad Minus', 'Decrease landmark size'),
    toggle_mesh =('space', 'Space', 'Toggle mesh visibility (to check landmark indices)'),
)


class Landmark:

    label_kwargs = dict(
        show_points=False, fill_shape=False, always_visible=True, margin=0, shape_opacity=0, font_size=21
    )

    def __init__(self, plotter, position, index, scale):
        self.plotter = plotter
        self.pos = position
        self.index = index
        self.radius = scale
        self.sphere = pv.Sphere(radius=1, center=self.pos, theta_resolution=10, phi_resolution=10)
        self.base_points = self.sphere.points - np.array(self.pos)[None, :]
        self.npy_pos = np.array(self.pos)[None, :]
        self.sphere.points = self.npy_pos + self.radius * self.base_points
        self.label_actor = self.plotter.add_point_labels(self.npy_pos, [str(self.index)], **Landmark.label_kwargs)
        self.set_color(LAST_COLOR)
        self.actor = self.plotter.add_mesh(self.sphere, scalars='color', rgb=True, lighting=False, reset_camera=False, show_scalar_bar=False)
        self.bad = False
    
    def set_color(self, color):
        color_array = np.array([color] * self.sphere.points.shape[0])
        self.sphere['color'] = color_array

    def __del__(self):
        if self.plotter.active:
            self.plotter.remove_actor(self.actor)
            self.plotter.remove_actor(self.label_actor)

    def move(self, position):
        delta = position - np.array(self.pos)
        self.pos = (position[0], position[1], position[2])
        self.sphere.points += delta[None, :]
        self.plotter.remove_actor(self.label_actor)
        self.npy_pos += delta[None, :]
        self.label_actor = self.plotter.add_point_labels(self.npy_pos, [str(self.index)], **Landmark.label_kwargs)
    
    def swap(l1, l2):
        l1_pos = np.array(l1.pos)
        l2_pos = np.array(l2.pos)
        l1.move(l2_pos)
        l2.move(l1_pos)

    def update_scale(self, s):
        self.radius = s
        center = np.array(self.pos)[None, :]
        self.sphere.points = s * self.base_points + center


def get_mesh_scale(mesh):
    min = np.min(mesh.vertices, axis=0)
    max = np.max(mesh.vertices, axis=0)
    return np.linalg.norm(max - min)

class Manager:
   
    def __init__(self, mesh, nb_landmarks):

        self.mesh = mesh
        self.pv_mesh = pv.wrap(mesh)
        self.landmarks = []
        self.nb_landmarks = nb_landmarks

        scale = get_mesh_scale(mesh)
        self.landmark_scale = LANDMARKS_SIZE * scale
        self.distance_treshold = DISTANCE_THRESHOLD * scale

        self.plotter = pvqt.BackgroundPlotter()
        self.plotter.set_icon('mesh_landmarking.jpg')

        self.plotter.iren.picker = 'world'
        self.plotter.background_color = 'w'
        self.actor = self.plotter.add_mesh(self.pv_mesh, show_edges=False, edge_color='black')
        self.plotter.camera.reset_clipping_range()
        
        self.plotter.key_press_event_signal.connect(self.process_key_press_event)
        self.plotter.track_mouse_position()
        def deactivate_plotter():
            self.plotter.active=False
            self.plotter.close()
        self.plotter.app_window.signal_close.connect(deactivate_plotter)
        self.selected_landmark = None

    def toggleMeshVisibility(self):
        visibility = self.actor.GetVisibility()
        self.actor.SetVisibility(not visibility)

    def _unselect_landmark(self):
        if self.selected_landmark is not None:
            color = LAST_COLOR if self.selected_landmark.index == len(self.landmarks) - 1 else BASE_COLOR
            self.selected_landmark.set_color(color)
            self.selected_landmark = None

    def add_landmark(self, position):
        if len(self.landmarks) < self.nb_landmarks:
            projected, dist, _ = tm.proximity.closest_point(self.mesh, np.array(position)[None, :])
            if dist[0] < self.distance_treshold :
                index = len(self.landmarks)
                ldm = Landmark(self.plotter, tuple(projected[0]), index, self.landmark_scale)
                ldm.update_scale(self.landmark_scale)
                self.landmarks.append(ldm)
            if len(self.landmarks) >= 2:
                self.landmarks[-2].set_color(BASE_COLOR)

    def pop_landmark(self):
        if len(self.landmarks) > 0:
            self.landmarks.pop()
            if len(self.landmarks) > 0:
                self.landmarks[-1].set_color(LAST_COLOR)
        
    def pick_landmark(self, position):
        if len(self.landmarks) == 0:
            return None
        pos = np.asarray(position)
        for landmark in reversed(self.landmarks):
            lpos = np.asarray(landmark.pos)
            dist = np.linalg.norm(lpos - pos)
            if dist < landmark.radius * 2:
                return landmark

    def move_landmark(self, landmark, position):
        if len(self.landmarks) == 0:
            return None
        pos = np.asarray(position)
        projected, dist, _ = tm.proximity.closest_point(self.mesh, pos[None, :])
        if dist[0] < self.distance_treshold :
            landmark.move(tuple(projected[0]))

    def set_landmarks(self, positional_landmarks):
        self.landmarks.clear()

        for i in range(positional_landmarks.shape[0]):
            pos = tuple(positional_landmarks[i, :3])

            ldm = Landmark(self.plotter, pos, i, self.landmark_scale)
            if i < len(positional_landmarks) - 1:
                ldm.set_color(BASE_COLOR)

            if len(positional_landmarks == 4):
                ldm.bad = positional_landmarks[i, 3] == 0.0
            if ldm.bad:
                ldm.set_color(BAD_COLOR)
            self.landmarks.append(ldm)
    
    def get_landmarks(self):
        positional_landmarks = np.zeros((len(self.landmarks), 4))
        positional_landmarks[:, 3] = 1.0
        for ldm in self.landmarks:
            index = ldm.index
            positional_landmarks[index, :3] = np.asarray(ldm.pos)
            positional_landmarks[index, 3] = float(not ldm.bad)
        return positional_landmarks

    def scale_landmarks(self, s):
        self.landmark_scale *= s
        for landmark in self.landmarks:
            landmark.update_scale(self.landmark_scale)

    def process_key_press_event(self, obj, _):
        self.changed = False
        code = obj.GetKeySym()

        if code == COMMANDS['remove_landmark'][0]:
            if self.selected_landmark is None or self.selected_landmark.index < len(self.landmarks) - 1:
                self.pop_landmark()
        elif code == COMMANDS['place_landmark'][0]:
            if self.selected_landmark is None:
                pos = self.plotter.pick_mouse_position()
                self.add_landmark(pos)
        elif code == COMMANDS['select_landmark'][0]:
            pos = self.plotter.pick_mouse_position()
            if self.selected_landmark is None:
                self.selected_landmark = self.pick_landmark(pos)
                if self.selected_landmark is not None:
                    self.selected_landmark.set_color(SELECTED_COLOR)
            else:
                self.move_landmark(self.selected_landmark, pos)
                self._unselect_landmark()
        elif code == COMMANDS['bad_landmark'][0]:
            pos = self.plotter.pick_mouse_position()
            ldm = self.pick_landmark(pos)
            if ldm is not None:
                ldm.bad = not ldm.bad
                if ldm.bad:
                    color = BAD_COLOR
                else:
                    color = LAST_COLOR if ldm.index == len(self.landmarks) - 1 else BASE_COLOR
                ldm.set_color(color)
        elif code == COMMANDS['unselect_landmark'][0]:
            self._unselect_landmark()
        elif code == COMMANDS['swap_landmarks'][0]:
            if self.selected_landmark is not None:
                pos = self.plotter.pick_mouse_position()
                other_landmark = self.pick_landmark(pos)
                if other_landmark is not None:
                    Landmark.swap(self.selected_landmark, other_landmark)
                    self._unselect_landmark()
        elif code == COMMANDS['check_landmark'][0]:
            pos = self.plotter.pick_mouse_position()
            ldm = self.pick_landmark(pos)
            if ldm is not None:
                print(f'Landmark index : {ldm.index}')
        elif code == COMMANDS['increase_landmark_size'][0]:
            self.scale_landmarks(SCALING_FACTOR)
        elif code == COMMANDS['decrease_landmark_size'][0]:
            self.scale_landmarks(1/SCALING_FACTOR)
        elif code == COMMANDS['toggle_mesh'][0]:
            self.toggleMeshVisibility()
        elif code == 'n':
            print(len(self.landmarks), 'landmarks')
            for ldm in self.landmarks:
                print(ldm.pos)
        elif code == 'q':
            print('Saving and exiting.')
        else:
            print(code, 'is not bound to any action.')
    

def print_keys():
    print('#' * 100)
    print('#' * 100)
    for action in COMMANDS:
        _, key, descr = COMMANDS[action]
        print('#\t' + key + ' :\t' + descr)
    print('#' * 100)
    print('#' * 100)


def barycentric_to_positional(mesh, landmarks):
    res = np.zeros((landmarks.shape[0], 4))
    if landmarks.shape[0] == 0:
        return res

    tids = landmarks[:, 0].astype(np.int32)
    barys = landmarks[:, 1:4]
    positions = tm.triangles.barycentric_to_points(mesh.triangles[tids], barys)
    
    res[:, 0:3] = positions
    res[:, 3] = landmarks[:, 4]
    return res

def positional_to_barycentric(mesh, landmarks):
    res = np.zeros((landmarks.shape[0], 5))
    if landmarks.shape[0] == 0:
        return res

    positions = landmarks[:, 0:3]
    projected, _, tids = tm.proximity.closest_point(mesh, positions)
    barys = tm.triangles.points_to_barycentric(mesh.triangles[tids], projected)
   
    res[:, 0] = tids.astype(np.float64)
    res[:, 1:4] = barys
    res[:, 4] = landmarks[:, 3]
    return res

@click.command()
@click.argument('mesh_path', type=click.Path(exists=True), required=True)
@click.argument('landmarks_path', type=click.Path(exists=False), required=True)
@click.option('-nl', '--nb_landmarks', type=int, default=sys.maxsize, help='Number of landmarks to place')
@click.option('-b', '--barycentric', is_flag=True, type=bool, help='If present, landmarks are saved in barycentric format.')
def cli(mesh_path, landmarks_path, nb_landmarks, barycentric):
    print_keys()
    mesh = tm.load(mesh_path, process=False)
    if os.path.exists(landmarks_path):
        landmarks = np.load(landmarks_path)
    else:
        landmarks = np.zeros((0, 4))
    
    if landmarks.shape[1] == 5:
        positional_landmarks = barycentric_to_positional(mesh, landmarks)
    else:
        positional_landmarks = landmarks
    
    manager = Manager(mesh, nb_landmarks)

    manager.set_landmarks(positional_landmarks)

    while manager.plotter.active:
        manager.plotter.app.processEvents()
    output_landmarks = manager.get_landmarks()

    if barycentric:
        bary_output_landmarks = positional_to_barycentric(mesh, output_landmarks)
        np.save(landmarks_path, bary_output_landmarks)
    else:
        np.save(landmarks_path, output_landmarks)

if __name__ == '__main__':
    cli()


