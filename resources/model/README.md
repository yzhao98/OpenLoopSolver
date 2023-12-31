# iiwa Models
There are several iiwa models in quoted repo, iiwa14 and iiwa7.
The iiwa14 model has two versions, one marked with `ETHZ` is the model redefined from information identified at ETHZ.
The other gazebo, transmission, materials and utilities xacro files are files shared by both iiwa14 and iiwa7.

See [configuration](../../src/manipulator/constants.py) for all models configured.

## iiwa14 ETHZ
⚠ This description file is not physical!!! We are dropping the usage of it. Its mesh file related to collision is also corrupted that drake cannot load it.
⚠

The definition of iiwa14_ETHZ is from the [repo](https://github.com/kuka-isir/iiwa_description) and generated by ROS tool `xacro`, a macro tool for URDF file.
The [iiwa14_ETHZ.urdf](iiwa_description/urdf/iiwa14_ETHZ.urdf) is generated from the original definition file [iiwa14_ETHZ.urdf.xacro](https://github.com/kuka-isir/iiwa_description/blob/master/urdf/iiwa14_ETHZ.urdf.xacro).
To generate the URDF file, one should either install the package as a ROS package, or modify the definitions as follows,
- iiwa14_ETHZ.urdf.xacro: the filename attributes use `$(find iiwa_description)` to find the ROS package.
  We may change it to parent directory `..` as we will be calling `xacro` in the directory containing the `.xacro` file.
- iiwa14_ETHZ.xacro: this file will be loaded as instructed by previous xacro file. It also uses the find syntax of ROS.
  We need to change all of them.
- For definition quoting the mesh file(which is used by visual model and collision model),
  e.g. `<mesh filename="package://iiwa_description/meshes/iiwa14/visual/link_7.stl"/>`,
  we do not need to change the directory. We will provide the package directory to the mesh loader so that they know where to find the mesh files.

## iiwa14 drake
The description is available [here](https://github.com/RobotLocomotion/drake/blob/master/manipulation/models/iiwa_description/urdf/iiwa14_no_collision.urdf).
If we compare drake's iiwa7 (as described below) and drake's iiwa14, we find that iiwa7 is a simpler description file that is more suitable for teaching a course.
In iiwa7, it defines only links (link0 to link7) and joints (joint1 to joint7) while 
in iiwa14 there are base link and a base joint (a fixed joint with pose=all-zero=identity) binding link0 with base link. 
The base link is actually a virtual link (no geometry, no kinematics parameters) acting as model's root.

Besides, unlink the iiwa14_ETHZ configuration, in this configuration, the end_effector link 
(which is a placeholder link without mass and inertia) is named,
- iiwa_link_ee_kuka
- iiwa_link_ee
 
They both fixed to iiwa_link_7 with different joints (that having different poses w.r.t. link_7).

### Weld the manipulator
For our application, we don't have a floating body. Namely, we need to weld link0 of iiwa7 and base of iiwa14 to the world frame.
This is automatically done in Pinocchio as it fix all bodies without mass.
In Drake, we have to do this manually. This can be done by [welding frames](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html#a51239c495a32c1f156964cd2c8410c53).

## iiwa7 drake
This frequently used model is only available in SDF format which is not supported by Pinocchio yet. 
The support of SDF will be available in next major version of Pinocchio, see [PreLease](https://github.com/stack-of-tasks/pinocchio/releases/tag/v2.9.1).
The converter 
(see [guide](https://ik-test.readthedocs.io/en/latest/pysdf/) and [code](https://github.com/andreasBihlmaier/pysdf), 
it can be easily installed by initializing a catkin workspace and run catkin_make there) does not support the SDF version specified in 
[drake iiwa7 description](https://github.com/RobotLocomotion/drake/blob/master/manipulation/models/iiwa_description/urdf/iiwa14_no_collision.urdf).

## Franka Panda
- panda_no_collision:
  - remove collision model
  - remove grippers (attached to hand)
  - remove safety controllers
  - add `.1` to `panda_link8` (which fixed to `panda_link7` and with `panda_hand` attached); the mass was `0.0` might due to typos.
  - add transmissions

- panda_simple_collision:
  - base: panda_no_collision
  - add simple balls collision model to all links
 