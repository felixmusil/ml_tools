all:
	+$(MAKE) -C $(shell pwd)/ml_tools/descriptors/radial_numerial_integration/

clean:
	+rm $(shell pwd)/ml_tools/descriptors/radial_numerial_integration/*.so