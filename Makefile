all:
	+$(MAKE) -C $(shell pwd)/ml_tools/descriptors/dvr_radial_basis/

clean:
	+rm $(shell pwd)/ml_tools/descriptors/dvr_radial_basis/*.so