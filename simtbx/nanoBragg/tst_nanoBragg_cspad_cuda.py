from __future__ import division
import numpy as np

from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import DetectorFactory
from dxtbx.model.crystal import CrystalFactory
from scitbx.matrix import sqr, col

import simtbx.nanoBragg
from simtbx.nanoBragg.test_data import dxtbx_cspad

FF = 3e4  # DEFAULT FORMFACTOR

def main():
    # define a crystal dxtbx
    cr = {'__id__': 'crystal',
          'real_space_a': (200, 0, 0),
          'real_space_b': (0, 180, 0),
          'real_space_c': (0, 0, 150),
          'space_group_hall_symbol': '-P 4 2'}
    cryst = CrystalFactory.from_dict(cr)

    # flat CSPAD dxtbx
    cspad = DetectorFactory.from_dict(dxtbx_cspad.cspad)  # loads a 64 panel dxtbx cspad
    # beam dxtbx along Zaxis by default
    beam = BeamFactory.simple(1.3)  #  make a simple beam along z

    Npan = len( cspad)  # 64 panels on cspad

    # iterate over each panel, instantiate nanoBragg, then cache the
    # parameters that are changing with each instantiation,
    # in this case slow fast origin vectors, beam center, detector distance 
    dists, close_dists, beam_cents = [],[],[]
    sdets, fdets,odets = [],[],[]
    for pid in range(Npan):
        SIM = simtbx.nanoBragg.nanoBragg(detector=cspad, beam=beam,
                                         verbose=0, panel_id=pid)
        dists.append( SIM.distance_mm)
        close_dists.append( SIM.close_distance_mm)
        beam_cents.append( SIM.beam_center_mm)
        sdets.append( SIM.sdet_vector)
        fdets.append( SIM.fdet_vector)
        odets.append( SIM.odet_vector)

    # ^ we will update these cached variables, and only these cached variabe, when we simulate 
    # each panel on the cuda kernel. We will also reset the float image to zeros 

    # in addition to the geom args, we will also simulate 20 Amatrices on the GPU
    Nrotations = 20
    Zaxis = col( (0,0,1))
    Zrots = [Zaxis.axis_and_angle_as_r3_rotation_matrix(deg,deg=True)
        for deg in np.linspace(-1,1,Nrotations)]

    # so our master simulation is:
    # for each A matrix in 20 Amatrices
    #    for each panel in 63 panels
    #        append the simulated panel
    panel_data = []
    import time
    t = time.time()
    for i_rot in range( Nrotations):
        print "<><><> %d <><><>" % i_rot
        for pid in range(Npan):
            SIM = simtbx.nanoBragg.nanoBragg(detector=cspad, beam=beam,
                verbose=0, panel_id=pid)

            B = sqr(cryst.get_B())
            U = sqr(cryst.get_U())
            Uz = Zrots[ i_rot]
            A = Uz*U*B
            SIM.Amatrix = A.transpose().elems
            SIM.Ncells_abc = (15, 15, 15)
            SIM.flux = 1e12
            SIM.mosaic_spread_deg = 0.02
            SIM.mosaic_domains = 10
            SIM.polarization = 1
            SIM.F000 = FF
            SIM.default_F = FF
            SIM.progress_meter = True
            SIM.beamsize_mm = 0.005
            SIM.exposure_s = 1
            SIM.Ncells_abc = (15, 15, 15)
            SIM.add_nanoBragg_spots_cuda()
            panel_data.append( SIM.raw_pixels.as_numpy_array())

    print "\n\n<><><><><><>\nOLD WAY TOOK %.3f seconds\n<><><><><><>" % (time.time() - t)

    # Now, instead of calling the old function add_nanoBragg_spots_cuda() 20 * 64 times which
    # updates all parameters, we will call the new function only updating the small variables
    # related to the geometry of the panel and crystal, and simply reset the float image to 0
    panel_data2 = []
    t = time.time()
    SIM = simtbx.nanoBragg.nanoBragg(detector=cspad, beam=beam,
                                     verbose=0, panel_id=pid)
    SIM.Amatrix = sqr(cryst.get_A()).transpose().elems
    SIM.flux = 1e12
    SIM.mosaic_spread_deg = 0.02
    SIM.mosaic_domains = 10
    SIM.polarization = 1
    SIM.F000 = FF
    SIM.default_F = FF
    SIM.progress_meter = True
    SIM.beamsize_mm = 0.005
    SIM.exposure_s = 1
    SIM.Ncells_abc = (15, 15, 15)
    SIM.allocate_cuda()
    for i_rot in range( Nrotations):
        for pid in range(Npan):

            B = sqr(cryst.get_B())
            U = sqr(cryst.get_U())
            Uz = Zrots[ i_rot]
            A = Uz*U*B
            SIM.Amatrix = A.transpose().elems
            SIM.sdet_vector = sdets[pid]
            SIM.fdet_vector = fdets[pid]
            SIM.odet_vector = odets[pid]
            SIM.beam_center_mm = beam_cents[pid]
            SIM.distance_mm = dists[pid]
            SIM.close_distance_mm = close_dists[pid]

            SIM.add_nanoBragg_spots_cuda_update()  # this resets the floatimage (rawpixels) to 0 and updates small data structures like 4-vectors
            SIM.get_raw_pixels_cuda()
            panel_data2.append( SIM.raw_pixels.as_numpy_array())

    SIM.deallocate_cuda()

    print "\n\n<><><><><><>\nNEW WAY TOOK %.3f seconds\n<><><><><><>" % (time.time() - t)
    #np.savez("data", p=panel_data, p2=panel_data2)
    for pid in range(64):
        assert( np.allclose(panel_data[pid], panel_data2[pid]))

if __name__=="__main__":
    main()
    print("OK")
