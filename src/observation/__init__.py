"""Turn an external galaxy catalogue (real KiDS-Legacy data, a blinded copy, or a saved mock catalogue)
into the HDF5 observation the trained models consume.

An observation has a mock's cls_results/ and pixelised_results/ groups and an empty cosmo_dict. The
package calls the protected src/cosmology functions in the production order and never reads,
infers or writes a cosmology. Import from the defining modules.
"""
