all:
	+$(MAKE) -C $(shell pwd)/ml_tools/descriptors/approx_dirac_radial_basis/

clean:
	+rm $(shell pwd)/ml_tools/descriptors/approx_dirac_radial_basis/*.so