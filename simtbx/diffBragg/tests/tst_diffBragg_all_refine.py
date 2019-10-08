from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--plot", action='store_true')
args = parser.parse_args()

from dxtbx.model.crystal import Crystal
from copy import deepcopy
from dxtbx.model import Panel
from cctbx import uctbx
from scitbx.matrix import sqr, rec, col
import numpy as np
from scipy.spatial.transform import Rotation
import pylab as plt

from simtbx.diffBragg.nanoBragg_crystal import nanoBragg_crystal
from simtbx.diffBragg.sim_data import SimData
from simtbx.diffBragg import utils
from simtbx.diffBragg.refiners import RefineAll
from simtbx.diffBragg.refiners.crystal_systems import MonoclinicManager

ucell = (55, 65, 75, 90, 95, 90)
ucell2 = (55.2, 65.2, 74.7, 90, 94.3, 90)
symbol = "P121"

# generate a random raotation
rotation = Rotation.random(num=1, random_state=1147)[0]
Q = rec(rotation.as_quat(), n=(4, 1))
rot_ang, rot_axis = Q.unit_quaternion_as_axis_and_angle()

# generate a small perturbation rotation
np.random.seed(1)
perturb_rot_axis = np.random.random(3)
perturb_rot_axis /= np.linalg.norm(perturb_rot_axis)
perturb_rot_ang = 0.1  # 0.1 degree random perturbtation

# make the ground truth crystal:
a_real, b_real, c_real = sqr(uctbx.unit_cell(ucell).orthogonalization_matrix()).transpose().as_list_of_lists()
C = Crystal(a_real, b_real, c_real, symbol)
C.rotate_around_origin(rot_axis, rot_ang)

a2_real, b2_real, c2_real = sqr(uctbx.unit_cell(ucell2).orthogonalization_matrix()).transpose().as_list_of_lists()
C2 = Crystal(a2_real, b2_real, c2_real, symbol)
C2.rotate_around_origin(rot_axis, rot_ang)
assert np.allclose(C2.get_U(), C.get_U())
C2.rotate_around_origin(col(perturb_rot_axis), perturb_rot_ang)

# Setup the simulation and create a realistic image
# with background and noise
# <><><><><><><><><><><><><><><><><><><><><><><><><>
nbcryst = nanoBragg_crystal()
nbcryst.dxtbx_crystal = C   # simulate ground truth
nbcryst.thick_mm = 0.1
nbcryst.Ncells_abc = 12, 12, 12  # ground truth Ncells
print("Ground truth ncells = %f" % (nbcryst.Ncells_abc[0]))

SIM = SimData()
SIM.detector = SimData.simple_detector(150, 0.1, (1024, 1024))

# TODO get the detector model
node = SIM.detector[0]
node_d = node.to_dict()
Origin = node_d["origin"][0], node_d["origin"][1], node_d["origin"][2]
distance = Origin[2]
print "Ground truth originZ=%f" % (SIM.detector[0].get_origin()[2])

# TODO perturb the detector model
# copy the detector and update the origin
det2 = deepcopy(SIM.detector)
# alter the detector distance by 2 mm
node_d["origin"] = Origin[0], Origin[1], Origin[2]+0.75
det2[0] = Panel.from_dict(node_d)
print ("Modified originZ=%f" % (det2[0].get_origin()[2]))

SIM.crystal = nbcryst
SIM.instantiate_diffBragg(oversample=0)
SIM.D.progress_meter = False
SIM.water_path_mm = 0.005
SIM.air_path_mm = 0.1
SIM.add_air = True
SIM.add_Water = True
SIM.include_noise = True
SIM.D.add_diffBragg_spots()
spots = SIM.D.raw_pixels.as_numpy_array()
SIM._add_background()
SIM._add_noise()
# This is the ground truth image:
img = SIM.D.raw_pixels.as_numpy_array()
SIM.D.raw_pixels *= 0

# Simulate the perturbed image for comparison
# perturbed detector:
SIM.detector = det2
SIM.D.update_dxtbx_geoms(det2, SIM.beam.nanoBragg_constructor_beam, 0)
# perturbed crystal:
SIM.D.Bmatrix = C2.get_B()
SIM.D.Umatrix = C2.get_U()
Ncells_abc2 = 15, 15, 15
nbcryst.dxtbx_crystal = C2
nbcryst.Ncells_abc = Ncells_abc2
SIM.crystal = nbcryst
# perturbed Ncells
SIM.D.set_value(9, Ncells_abc2[0])
SIM.D.add_diffBragg_spots()
SIM._add_background()
SIM._add_noise()
# Perturbed image:
img_pet = SIM.D.raw_pixels.as_numpy_array()
SIM.D.raw_pixels *= 0

# spot_rois, abc_init , these are inputs to the refiner
# <><><><><><><><><><><><><><><><><><><><><><><><><>
spot_roi, tilt_abc = utils.process_simdata(spots, img, thresh=20, plot=args.plot)

UcellMan = MonoclinicManager(
    a=ucell2[0],
    b=ucell2[1],
    c=ucell2[2],
    beta=ucell2[4]*np.pi/180.)

init_Umat_norm = np.abs(np.array(C2.get_U()) - np.array(C.get_U())).sum()
init_Bmat_norm = np.abs(np.array(C2.get_B()) - np.array(C.get_B())).sum()

RUC = RefineAll(
    spot_rois=spot_roi,
    abc_init=tilt_abc,
    img=img,
    SimData_instance=SIM,
    plot_images=args.plot,
    plot_residuals=True,
    ucell_manager=UcellMan)
RUC.trad_conv = True
RUC.refine_detdist = True
RUC.refine_Amatrix = True
RUC.refine_ncells = True
RUC.trad_conv_eps = 1e-5
RUC.max_calls = 2000000
RUC.run()

ang, ax = RUC.get_correction_misset(as_axis_angle_deg=True)
C2.rotate_around_origin(ax, ang)
C2.set_B(RUC.get_refined_Bmatrix())

final_Umat_norm = np.abs(np.array(C2.get_U()) - np.array(C.get_U())).sum()
final_Bmat_norm = np.abs(np.array(C2.get_B()) - np.array(C.get_B())).sum()

# refined unit cell parameters
ucell_ref = C2.get_unit_cell().parameters()

print("Results!")
print("Before refinement: Umatrix distance=%2.7g, Bmatrix distance=%2.7g" % (init_Umat_norm, init_Bmat_norm))
print("After refinement: Umatrix distance=%2.7g, Bmatrix distance=%2.7g" % (final_Umat_norm, final_Bmat_norm))
print("")
print("ground truth unit cell: %2.7g,%2.7g,%2.7g,%2.7g,%2.7g,%2.7g" % ucell)
print("unit cell passed to refinement: %2.7g,%2.7g,%2.7g,%2.7g,%2.7g,%2.7g" % ucell2)
print("refined unit cell: %2.7g,%2.7g,%2.7g,%2.7g,%2.7g,%2.7g" % ucell_ref)
print("")
print("Perturbation axis =%+2.7g,%+2.7g,%+2.7g and angle=%+2.7g deg"
      % (perturb_rot_axis[0], perturb_rot_axis[1], perturb_rot_axis[2], perturb_rot_ang))
print("Misset applied during refinement: axis=%+2.7g,%+2.7g,%+2.7g and angle=%+2.7g deg"
      % (ax[0], ax[1], ax[2], ang))

# error in initial unit cell parameters
err_init = np.linalg.norm([abs(u-u_init)/u for u, u_init in zip(ucell, ucell2)])*100

# error in refined unit cell parameters
err_ref = np.linalg.norm([abs(u-u_ref)/u for u, u_ref in zip(ucell, ucell_ref)])*100

assert err_ref < 1e-1 * err_init

# Note, this test might change, e.g. angle could be negative and axis could be the same...
assert np.round(ang, 1) == np.round(perturb_rot_ang, 1)
assert np.linalg.norm(np.round(ax, 2) + np.round(perturb_rot_axis, 2)) < 0.075
assert final_Umat_norm < 1e-1*init_Umat_norm

print("OK")
