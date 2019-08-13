"""
call this test with 1 argument that is either
  cpu, gpu, VS

Note, VS will fail currently
Note, cpu does not require gpu funcionality, but gpu and VS do
"""

import os
import numpy as np
import h5py
from scipy import constants
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt

from simtbx.nanoBragg import shapetype
from dxtbx_model_ext import flex_Beam
from simtbx.nanoBragg import nanoBragg
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from scitbx.matrix import sqr

# load a pdb file from iotbx for gen of structure facto
import iotbx
ipath = os.path.dirname(iotbx.__file__)
pdbpath = os.path.join(ipath, "regression/secondary_structure/1ubf_cutted.pdb")

# make a crystal dxtbx model
cr = {'__id__': 'crystal',
      'real_space_a': (80, 0, 0),
      'real_space_b': (0, 80, 0),
      'real_space_c': (0, 0, 40),
      'space_group_hall_symbol': '-P 4 2'}
cryst = CrystalFactory.from_dict(cr)

# nanoBragg property values as globals
AMAT = sqr(cryst.get_A()).transpose().elems
DEFAULT_F = 0
NCELLS_ABC = (15,15,15)
DET_SHAPE = (1024,1024)
POLA = 1
ROI = None
verbose=1 

# define some beams
Nbeams = 128
en_min = 8540
en_max = 9160
energies = np.linspace(en_min, en_max ,Nbeams)  # 1 eV beam separation
wavelens = ENERGY_CONV / energies
ave_flux=1e12
exposure_s = 1
np.random.seed(0)
fluxes = np.random.uniform(0.5, 1.5, Nbeams) * ave_flux / exposure_s
print("Fluxes", fluxes)

# Generate the Fhkl at each wavelength 
from simtbx.nanoBragg.tst_nanoBragg_basic import fcalc_from_pdb
multi_source_Fhkl = []
for i in range(Nbeams):
    print ("Generating structure factors for beam %d / %d" % (i+1, Nbeams))
    F = fcalc_from_pdb(resolution=3, algorithm="fft", wavelength=wavelens[i], pdbin=pdbpath)
    multi_source_Fhkl.append(F)

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# the goal is to simulate using N  1-beam instantiations of simtbx
# and then try to do the same simulation using 1 N-beam instantion of simtbx
# the results should be the same
# This makes use of the internal loop of sources within nanoBragg
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def test_xraybeams_laue(shape=shapetype.Square,cudaA=False, cudaB = False):
    """
    :param shape: nanoBragg crystal shape (nanoBragg.shapetype.Gauss, .Tophat, .Round, or .Square)
    :param cudaA, cudaB: whether to use cuda kernel in the stepwise (A) or aggregate (B) computation
    :return:
    """

    # make a model beam
    beam_descr = {'direction': (-1, 0, 0),
                  'divergence': 0.0,
                  'flux': 1e12, 
                  'wavelength': 1e-10}   # overwrite wavelength and flux later

    xrbeams = flex_Beam()  # this can be appended to, right now its an empty beams list

    # stepwise is the method involving N 1-beam instantiations of nanoBragg
    
    import time
    stepwise_spots = np.zeros(DET_SHAPE)
    stepwise_time = 0
    for i_wav, (wav, fl) in enumerate(zip( wavelens, fluxes)):

        nbr = nanoBragg(detpixels_slowfast=DET_SHAPE, verbose=verbose, oversample=0)
        # nbr.raw_pixels is initialized as zeros
        if ROI is not None:
            nbr.region_of_interest = ROI
        #nbr.default_F = DEFAULT_F
        #nbr.F000 = 0 #DEFAULT_F
        nbr.xtal_shape = shape
        nbr.Ncells_abc = NCELLS_ABC
        nbr.wavelength_A = wav
        nbr.flux = fl
        nbr.polarization = POLA
        nbr.exposure_s = exposure_s
        nbr.progress_meter=True
        nbr.Fhkl = multi_source_Fhkl[i_wav]
        t = time.time()
        if cudaA:
            nbr.add_nanoBragg_spots_cuda()
        else:
            nbr.add_nanoBragg_spots()
        stepwise_spots += nbr.raw_pixels.as_numpy_array()   
        stepwise_time += time.time()-t

        # keep track of beams for single call to nanoBragg using xray_beams
        beam = BeamFactory.from_dict(beam_descr)
        beam.set_wavelength(wav*1e-10)  # need to fix the necessity to do this..
        beam.set_flux(fl)
        beam.set_direction((-1, 0, 0))   # this is the convention, stick with it
        xrbeams.append(beam)

    # Now do the 1 N-beams instantiaion of nanoBragg, using the flex beams object defined above
    print("\t<><><><><><><><><><")
    print("\tMULTI SOURCE TIME")
    print("\t<><><><><><><><><><")
    aggregate_time = 0
    nbr = nanoBragg(detpixels_slowfast=DET_SHAPE, verbose=verbose, oversample=0)
    if ROI is not None:
        nbr.region_of_interest = ROI
    nbr.xtal_shape = shape
    #nbr.default_F = DEFAULT_F
    nbr.polarization = POLA
    nbr.Ncells_abc = NCELLS_ABC 
    nbr.xray_beams = xrbeams  # set the beams@
    nbr.exposure_s = exposure_s
    nbr.Multisource_Fhkl = multi_source_Fhkl
    #nbr.F000 =  0#DEFAULT_F
    
    #nbr.default_F = DEFAULT_F
    #nbr.F000 = DEFAULT_F
    
    t = time.time()
    if cudaB:  # TODO: test the cuda kernel sources mode, likely will need patches like the CPU one did
        nbr.add_nanoBragg_spots_cuda()
    else:    
        nbr.add_nanoBragg_spots()
    aggregate_spots = nbr.raw_pixels.as_numpy_array()
    aggregate_time = time.time()-t
    assert(np.allclose(stepwise_spots, aggregate_spots, atol=0.5))

    return (stepwise_time, aggregate_time)

if __name__ == "__main__":
    import sys
    import time

    # do the LAUE test for each of the shape models and assert it passes
    if sys.argv[1] == "cpu":
        t = time.time()
        sq, sq2 = test_xraybeams_laue(shape=shapetype.Square)  # PASSES
        ga, ga2 = test_xraybeams_laue(shape=shapetype.Gauss) # PASSES
        rn, rn2 = test_xraybeams_laue(shape=shapetype.Round) # PASSES
        #test_xraybeams_laue(shape=shapetype.Tophat)  # FIXME: tophat is breaking!
        print "\nCPU RESULTS:"
        print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
        print      "Square model: N 1beam calls: %.3f   1 N-beam call: %.3f" % (sq, sq2)
        print      "Gauss model:  N 1beam calls: %.3f   1 N-beam call: %.3f" % (ga, ga2)
        print      "Round model:  N 1beam calls: %.3f   1 N-beam call: %.3f" % (rn, rn2)
        print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
        print

    elif sys.argv[1] == "gpu":
        t = time.time()
        sq, sq2 = test_xraybeams_laue(shape=shapetype.Square, cudaA=True, cudaB=True)
        ga, ga2 = test_xraybeams_laue(shape=shapetype.Gauss, cudaA=True, cudaB=True)
        rn, rn2 = test_xraybeams_laue(shape=shapetype.Round, cudaA=True, cudaB=True)
        print "\nGPU RESULTS (seconds):"
        print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
        print      "Square model: N 1beam calls: %.2e   1 N-beam call: %.2e" % (sq, sq2)
        print      "Gauss model:  N 1beam calls: %.2e   1 N-beam call: %.2e" % (ga, ga2)
        print      "Round model:  N 1beam calls: %.2e   1 N-beam call: %.2e" % (rn, rn2)
        print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
        print
    
    elif sys.argv[1] == "VS":  # FIXME: this is failing for all shapes and permutations, but its actually pretty close to passing!
        t = time.time()
        #test_xraybeams_laue(shape=shapetype.Round, cudaA=True, cudaB=False)
        test_xraybeams_laue(shape=shapetype.Square, cudaA=True, cudaB=False)
        #test_xraybeams_laue(shape=shapetype.Gauss, cudaA=True, cudaB=False)
        
        #test_xraybeams_laue(shape=shapetype.Round, cudaA=False, cudaB=True)
        test_xraybeams_laue(shape=shapetype.Square, cudaA=False, cudaB=True)
        #test_xraybeams_laue(shape=shapetype.Gauss, cudaA=False, cudaB=True)
        print "Took %.4f seconds" % (time.time() - t)

    print("OK")

