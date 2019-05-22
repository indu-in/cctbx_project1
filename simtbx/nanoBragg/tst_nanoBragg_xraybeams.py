"""
call this test with 1 argument that is either
  cpu, gpu, VS

Note, VS will fail currently
Note, cpu does not require gpu funcionality, but gpu and VS do
"""
from simtbx.nanoBragg import shapetype
from dxtbx_model_ext import flex_Beam
from simtbx.nanoBragg import nanoBragg
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from scitbx.matrix import sqr
import numpy as np

# make a crystal dxtbx model
cr = {'__id__': 'crystal',
      'real_space_a': (300, 0, 0),
      'real_space_b': (0, 200, 0),
      'real_space_c': (0, 0, 150),
      'space_group_hall_symbol': '-P 4 2'}
cryst = CrystalFactory.from_dict(cr)

# nanoBragg property values as globals
AMAT = sqr(cryst.get_A()).transpose().elems
DEFAULT_F = 1e7
NCELLS_ABC = (15,15,15)
DET_SHAPE = (256,256)
POLA = 1
ROI = None
#ROI = ((90,92),(105,107))  # NOTE can use ROI to speed up, but pretty fast as is, not sure how GPU code tests using ROI
#ROI = ((90,105),(105,120))

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
    Nbeams = 10  # number of beams (channels)
    banwd = 0.04  # xray bandwidth
    wavelens = np.linspace(
        1.2 - 1.2*banwd/2,
        1.2 + 1.2*banwd/2,
        Nbeams)
    np.random.seed(1)
    fluxes = np.random.uniform(0.5, 1.5, Nbeams) * 1e12
    

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
    for wav, fl in zip( wavelens, fluxes):
        t = time.time()

        nbr = nanoBragg(detpixels_slowfast=DET_SHAPE, verbose=10, oversample=0)
        # nbr.raw_pixels is initialized as zeros
        if ROI is not None:
            nbr.region_of_interest = ROI
        nbr.default_F = DEFAULT_F
        nbr.F000 = DEFAULT_F
        nbr.xtal_shape = shape
        nbr.Ncells_abc = NCELLS_ABC
        nbr.wavelength_A = wav
        nbr.flux = fl
        nbr.polarization = POLA
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
    aggregate_time = 0
    t = time.time()
    nbr = nanoBragg(detpixels_slowfast=DET_SHAPE, verbose=10, oversample=0)
    if ROI is not None:
        nbr.region_of_interest = ROI
    nbr.xtal_shape = shape
    nbr.default_F = DEFAULT_F
    nbr.F000 = DEFAULT_F
    nbr.polarization = POLA
    nbr.Ncells_abc = NCELLS_ABC 
    nbr.xray_beams = xrbeams  # set the beams@
    if cudaB:  # TODO: test the cuda kernel sources mode, likely will need patches like the CPU one did
        nbr.add_nanoBragg_spots_cuda()
    else:    
        nbr.add_nanoBragg_spots()
    aggregate_spots = nbr.raw_pixels.as_numpy_array()
    aggregate_time = time.time()-t

    #np.savez("_interps", agg=aggregate_spots, step=stepwise_spots)
    assert(np.allclose(stepwise_spots, aggregate_spots))

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
        print "\nGPU RESULTS:"
        print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
        print      "Square model: N 1beam calls: %.2e   1 N-beam call: %.2e" % (sq, sq2)
        print      "Gauss model:  N 1beam calls: %.2e   1 N-beam call: %.2e" % (ga, ga2)
        print      "Round model:  N 1beam calls: %.2e   1 N-beam call: %.2e" % (rn, rn2)
        print "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
        print
    
    elif sys.argv[1] == "VS":  # FIXME: this is failing for all shapes and permutations, but its actually pretty close to passing!
        t = time.time()
        #test_xraybeams_laue(shape=shapetype.Square, cudaA=True, cudaB=False)
        test_xraybeams_laue(shape=shapetype.Square, cudaA=False, cudaB=True)
        print "Took %.4f seconds" % (time.time() - t)

    print("OK")

