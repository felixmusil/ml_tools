COMP=f2py

ge: gaussian_expansion.f90
	$(COMP) -c $^ -m $@

clean:
	rm *.so
