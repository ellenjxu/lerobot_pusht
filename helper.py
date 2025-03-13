"""
During inference, we map from real -> sim -> real

Helper functions:

1. Image processing: image -> T (x,y,theta)
2. Forward kinematics: 6 joint angles -> calculate (x,y) in sim
3. Inverse kinematics: (x,y) -> 6 joint angles
"""

def process_image(img):
  """
  Convert image to (x,y,theta)
  """
  pass

def forward_kinematics():
  """
  Calculate (x,y) in sim from 6 joint angles
  """
  pass

def inverse_kinematics():
  """
  Calculate 6 joint angles from (x,y) in sim
  """
  pass

# assuem we have x,y,x_t,y_t,theta_t in the cartesian coord system
# TODO: scale up the T in the sim because right now factor of 24x instead of 32x
# but maybe it's okay if our end effector is bigger in the sim
def real2sim_transform(x,y,x_t,y_t,theta_t):
    w,h = 512, 512
    scale = 32
    x, y = x*scale, h - y*scale
    x_t, y_t = x_t*scale, h - y_t*scale
    theta_t = theta_t
    return x,y,x_t,y_t,theta_t