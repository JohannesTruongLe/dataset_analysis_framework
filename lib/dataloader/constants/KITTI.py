"""Dataloader constants."""

# KITTI label rows
TYPE = 'type'
TRUNCATED = 'truncated'
OCCLUDED = 'occluded'
ALPHA = 'alpha'
X_MIN = 'x_min'
X_MAX = 'x_max'
Y_MIN = 'y_min'
Y_MAX = 'y_max'
SIZE_X = 'size_x'
SIZE_Y = 'size_y'
SIZE_Z = 'size_z'
LOCATION_X = 'location_x'
LOCATION_Y = 'location_y'
LOCATION_Z ='location_z'
ROTATION_Y = 'rotation_y'

# List of cols in KITTI # TODO make me class attribute
KITTI_COLS = [TYPE,
              TRUNCATED,
              OCCLUDED,
              ALPHA,
              X_MIN,
              Y_MIN,
              X_MAX,
              Y_MAX,
              SIZE_X,
              SIZE_Y,
              SIZE_Z,
              LOCATION_X,
              LOCATION_Y,
              LOCATION_Z,
              ROTATION_Y]
