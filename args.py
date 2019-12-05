import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("-model", type=str, default="XY", help="interacting model [XY]")
parser.add_argument("-tau", type=float, default=0.025, help="initial inverse temperature [0.025]")
parser.add_argument("-NBeta", type=int, default=10, help="number of temperature point [10]")

parser.add_argument("-D", type=int, default=20, help="bond dimensions [20]")
parser.add_argument("-depth", type=int, default=3, help="sweep depth [3]")
parser.add_argument("-Niter", type=int, default=10, help="max Iteration of single isometry optimization [10]")
parser.add_argument("-Nsweep", type=int, default=3, help="number of sweeps [3]")
parser.add_argument("-opti", type=int, default=1, help="optimization flag [1]")

parser.add_argument("-use_float32", action="store_true", help="use float32")
parser.add_argument("-cuda", type=int, default=-1, help="GPU ID [default:-1 (CPU)] ")
parser.add_argument("-rdir", type=str, default="./Result/", help="result folder")

args = parser.parse_args()

print('---------------+---------')
print('%15s|%8s'%('arguments','values'))
print('---------------+---------')
for arg in vars(args):
    print('%15s|%8s'%(arg, getattr(args, arg)))
print('---------------+---------')
print('\n')
