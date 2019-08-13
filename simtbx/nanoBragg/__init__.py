from __future__ import absolute_import, division, print_function
import boost.python
import cctbx.uctbx # possibly implicit
from scitbx.array_family import flex
ext = boost.python.import_ext("simtbx_nanoBragg_ext")
from simtbx_nanoBragg_ext import *

@boost.python.inject_into(ext.nanoBragg)
class _():

  def __getattr__(self,name):
    """assemble miller array of structure factors used to compute spot intensities from the internal C cube array
       how do we specify docstrings for individual overriden members? """
    from cctbx.crystal import symmetry
    from cctbx.miller import set, array
    if name in ["Fhkl", "Multisource_Fhkl"]:
      cs = symmetry(unit_cell = self.unit_cell_Adeg,space_group="P 1")
      indices,data = self.Fhkl_tuple
      mset = set(crystal_symmetry=cs, anomalous_flag=True, indices=indices)
      #if name=="Fhkl":
      # TODO return all sources Fhkl ?
      return array(mset, data=data[:len(indices)]).set_observation_type_xray_amplitude()
      #else:
      #
      #  return

  def __setattr__(self,name,value):
    """use a P1 anomalous=True miller array to initialize the internal C cube array with structure factors for the spot intensities
       how do we specify docstrings for individual overriden members? """

    if name =="Multisource_Fhkl":
        print("Here in beams")
        try:
            Nbeams = self.xray_beams.size()
        except AttributeError:
            print("Cannot set multisource Fhkl as it requires Xray beams to be set. ")
        if Nbeams==1:
            print("Cannot set multisource Fhkls because there is only one source  xray beam!")
        elif Nbeams != len(value):
            print ("Multi source Fhkl array of length %d should be same as # of Xray beams (%d)"\
                 % (len(value), Nbeams ))
        else:
            Nsources = len(value)
            data = []
            for ii,ff in enumerate( value): #source_Farray in enumerate(value):
                if not ff.space_group_info().type().hall_symbol()==" P 1":
                    ff = ff.expand_to_p1()
                ff = ff.generate_bijvoet_mates()
                self.unit_cell_Adeg = ff.unit_cell()
                indices = ff.indices()
                data.append(ff.data())

            data = flex.double([amp for source_data in data for amp in source_data])
            self.multi_sources_Fhkl = True
            self.Fhkl_tuple = (indices, data)

    elif name =="Fhkl":
      value=value.expand_to_p1()
      value=value.generate_bijvoet_mates()
      assert value.space_group_info().type().lookup_symbol() == "P 1"
      # handle exception by expanding to P1
      assert value.anomalous_flag() == True
      # handle exception by copying all F(hkl) to F(-h-k-l)
      #assert values are amplitudes # not sure how to guarantee this
      self.unit_cell_Adeg = value.unit_cell()
      #self.mock_up_group = value.space_group()
      #self.mock_up_anomalous_flag = value.anomalous_flag()
      self.Fhkl_tuple = (value.indices(),value.data())
      self.multi_sources_Fhkl = False  # TODO whats a better name, maybe multi_source_Fhkl
    else:
      super(ext.nanoBragg,self).__setattr__(name,value)

  def to_smv_format_py(self,fileout,intfile_scale=0.0,debug_x=-1,debug_y=-1,
    rotmat=False,extra=None,verbose=False,gz=False):

    byte_order = "little_endian";

    #recast the image file write to Python to afford extra options: rotmat, extra, gz
    if gz:
      from libtbx.smart_open import for_writing
      outfile = for_writing(file_name=fileout+".gz", gzip_mode="wb")
    else:
      outfile = open(fileout,"w");

    outfile.write("{\nHEADER_BYTES=1024;\nDIM=2;\nBYTE_ORDER=%s;\nTYPE=unsigned_short;\n"%byte_order);
    outfile.write("SIZE1=%d;\nSIZE2=%d;\nPIXEL_SIZE=%g;\nDISTANCE=%g;\n"%(
      self.detpixels_fastslow[0],self.detpixels_fastslow[1],self.pixel_size_mm,self.distance_mm));
    outfile.write("WAVELENGTH=%g;\n"%self.wavelength_A);
    outfile.write("BEAM_CENTER_X=%g;\nBEAM_CENTER_Y=%g;\n"%self.beam_center_mm);
    outfile.write("ADXV_CENTER_X=%g;\nADXV_CENTER_Y=%g;\n"%self.adxv_beam_center_mm);
    outfile.write("MOSFLM_CENTER_X=%g;\nMOSFLM_CENTER_Y=%g;\n"%self.mosflm_beam_center_mm);
    outfile.write("DENZO_X_BEAM=%g;\nDENZO_Y_BEAM=%g;\n"%self.denzo_beam_center_mm);
    outfile.write("DIALS_ORIGIN=%g,%g,%g\n"%self.dials_origin_mm);
    outfile.write("XDS_ORGX=%g;\nXDS_ORGY=%g;\n"%self.XDS_ORGXY);
    outfile.write("CLOSE_DISTANCE=%g;\n"%self.close_distance_mm);
    outfile.write("PHI=%g;\nOSC_START=%g;\nOSC_RANGE=%g;\n"%(self.phi_deg,self.phi_deg,self.osc_deg));
    outfile.write("TIME=%g;\n"%self.exposure_s);
    outfile.write("TWOTHETA=%g;\n"%self.detector_twotheta_deg);
    outfile.write("DETECTOR_SN=000;\n");
    outfile.write("ADC_OFFSET=%g;\n"%self.adc_offset_adu);
    outfile.write("BEAMLINE=fake;\n");
    if rotmat:
      from scitbx.matrix import sqr
      RSABC = sqr(self.Amatrix).inverse().transpose()
      outfile.write("DIRECT_SPACE_ABC=%s;\n"%(",".join([repr(a) for a in RSABC.elems])))
    if extra is not None:
      outfile.write(extra)
    outfile.write("}\f");
    assert outfile.tell() < 1024, "SMV header too long, please edit this code and ask for more bytes."
    while ( outfile.tell() < 1024 ): outfile.write(" ")
    from boost.python import streambuf
    self.to_smv_format_streambuf(streambuf(outfile),intfile_scale,debug_x,debug_y)

    outfile.close();
