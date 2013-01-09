from __future__ import division
from libtbx import test_utils
import libtbx.load_env

tst_list = (
  "$D/rotamer/tst_rotamer_eval.py",
  "$D/monomer_library/tst_idealized_aa.py",
  "$D/regression/tst_ml_estimate.py",
  "$D/density_modification/tst_density_modification.py",
  "$D/geometry_restraints/tst_ramachandran.py",
  "$D/tst_map_type_parser.py",
  "$D/rsr/tst.py",
  "$D/polygon/tst.py",
  "$D/polygon/tst_gui.py",
  "$D/chemical_components/tst.py",
  "$D/regression/tst_add_h_to_water.py",
  "$D/rotamer/rotamer_eval.py",
  "$D/wwpdb/tst_standard_geometry_cif.py",
  "$D/tst_pdbtools.py",
  "$D/real_space/tst.py",
  "$D/ias/tst_ias.py",
  "$D/refinement/tst_fit_rotamers.py",
  ["$D/refinement/tst_anomalous_scatterer_groups.py", "P3"],
  "$D/refinement/tst_rigid_body.py",
  "$D/refinement/tst_rigid_body_groups_from_pdb_chains.py",
  "$D/refinement/tst_refinement_flags.py",
  "$D/torsion_restraints/tst_reference_model.py",
  "$D/tst_model.py",
  "$D/tst_fmodel.py",
  "$D/tst_utils.py",
  "$D/tst_alignment.py",
  ["$D/tst_fmodel_fd.py", "P31"],
  "$D/ncs/tst_restraints.py",
  ["$D/ncs/ncs.py", "exercise"],
  "$D/regression/tst_adp_restraints.py",
  "$D/regression/tst_validate_utils.py",
  "$D/scaling/tst_scaling.py",
  "$D/scaling/tst_outlier.py",
  "$D/scaling/matthews.py",
  "$D/scaling/absence_likelihood.py",
  ["$D/scaling/thorough_outlier_test.py", "P21"],
  "$D/twinning/probabalistic_detwinning.py",
  "$D/monomer_library/tst_rna_sugar_pucker_analysis.py",
  "$D/monomer_library/tst_cif_types.py",
  "$D/monomer_library/tst_motif.py",
  "$D/monomer_library/tst_cif_triage.py",
  "$D/monomer_library/tst_rotamer_utils.py",
  "$D/monomer_library/tst_selection.py",
  "$D/monomer_library/tst_tyr_from_gly_and_bnz.py",
  "$D/monomer_library/tst_pdb_interpretation.py",
  "$D/monomer_library/tst_rna_dna_interpretation.py",
  "$D/monomer_library/tst_protein_interpretation.py",
  "$D/monomer_library/tst_geo_reduce_for_tardy.py",
  "$D/monomer_library/tst_linking.py",
  "$D/monomer_library/tst_neutron_distance.py",
  "$D/regression/tst_altloc_chain_break.py",
  "$D/hydrogens/build_hydrogens.py",
  "$D/max_lik/tst_maxlik.py",
  "$D/masks/tst_masks.py",
  "$D/masks/tst_asu_mask.py",
  "$D/max_lik/tst_max_lik.py",
  "$D/dynamics/tst_cartesian_dynamics.py",
  "$D/dynamics/tst_sa.py",
  "$D/tls/tst_tls.py",
  "$D/tls/tst_get_t_scheme.py",
  "$D/tls/tst_tls_refinement_fft.py",
  "$D/examples/f_model_manager.py",
  "$D/bulk_solvent/tst_bulk_solvent_and_scaling.py",
  "$D/bulk_solvent/tst_scaler.py",
  "$D/alignment.py",
  "$D/invariant_domain.py",
  "$D/secondary_structure/tst.py",
  "$D/geometry_restraints/tst_hbond.py",
  "$D/geometry_restraints/tst_reference_coordinate.py",
  "$D/conformation_dependent_library/test_cdl.py",
  "$D/validation/sequence.py",
  "$D/regression/tst_prune_model.py",
  "$D/regression/tst_real_space_correlation.py",
  "$D/regression/tst_examples.py",
  "$D/regression/tst_sort_hetatms.py",
  #
  "$D/refinement/real_space/tst_fit_residue_1.py",
  "$D/refinement/real_space/tst_fit_residue_2.py",
  "$D/refinement/real_space/tst_fit_residue_3.py",
  "$D/refinement/real_space/tst_fit_residue_4.py",
  "$D/refinement/real_space/tst_fit_residues_1.py",
  "$D/refinement/real_space/tst_fit_residues_2.py",
  "$D/refinement/real_space/tst_individual_sites_1.py",
  "$D/refinement/real_space/tst_monitor.py",
  "$D/refinement/real_space/tst_rigid_body.py",
  "$D/regression/tst_dssp.py",
  "$D/building/tst.py",
  "$D/building/disorder/tst.py",
  "$D/regression/tst_validation_summary.py",
  "$D/regression/tst_fmodel_twin_law.py",
  "$D/regression/tst_model_vs_data_twinned.py",
  )

def run():
  build_dir = libtbx.env.under_build("mmtbx")
  dist_dir = libtbx.env.dist_path("mmtbx")
  test_utils.run_tests(build_dir, dist_dir, tst_list)

if (__name__ == "__main__"):
  run()
