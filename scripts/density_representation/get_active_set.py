import argparse
import time

import sys
sys.path.insert(0,'../../')

import ml_tools as ml 
from ml_tools.base import np,sp



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Does a sample fps selection""")

    # parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    # parser.add_argument("-n", type=int, default=8, help="Number of radial functions for the descriptor")
    # parser.add_argument("-l", type=int, default=6, help="Maximum number of angular functions for the descriptor")
    # parser.add_argument("-c", type=float, default=3.5, help="Radial cutoff")
    # parser.add_argument("-cotw", type=float, default=0.5, help="Cutoff transition width")
    # parser.add_argument("-g", type=float, default=0.5, help="Atom Gaussian sigma")
    # parser.add_argument("-cw", type=float, default=1.0, help="Center atom weight")
    # parser.add_argument("-fa", "--fast-average", action='store_true', help="Fast average (soap vector are averaged over the frame in quippy -> less memory/computation intensive)")
    # parser.add_argument("-cp", "--chemical-projection", type=str, default="",
    #                     help="Filename of the chemical projection pickle.")
    # parser.add_argument("-k","--kernel", type=str, default="average",
    #                     help="Global kernel mode (e.g. --kernel average / rematch ")
    # parser.add_argument("-gm","--gamma", type=float, default=1.0,
    #                     help="Regularization for entropy-smoothed best-match kernel")
    # parser.add_argument("-z", "--zeta", type=int, default=2, help="Power for the environmental matrix")
    # parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
    # parser.add_argument("--first", type=int, default='0', help="Index of first frame to be read in")
    # parser.add_argument("--last", type=int, default='0', help="Index of last frame to be read in")
    # parser.add_argument("--outformat", type=str, default='text', help="Choose how to dump the alchemySoaps, e.g. pickle (default) or text (same as from glosim --verbose)")
    # parser.add_argument("-nt","--nthreads", type=int, default=4, help="Number of threads (1,2,4,6,9,12,16,25,36,48,64,81,100).")
    # parser.add_argument("-np","--nprocess", type=int, default=4, help="Number of processes to run in parallel.")
    # parser.add_argument("-nc","--nchunks", type=int, default=4, help="Number of chunks to divide the global kernel matrix in.")
    # parser.add_argument("--nocenters", type=str, default="",help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
    # parser.add_argument("-ngk","--normalize-global-kernel", action='store_true', help="Normalize global kernel")
    # parser.add_argument("-sek", "--save-env-kernels", action='store_true', help="Save environmental kernels")
    # parser.add_argument("-lm", "--low-memory", action='store_true', help="Computes the soap vectors in each thread when nchunks > 1")

    args = parser.parse_args()

