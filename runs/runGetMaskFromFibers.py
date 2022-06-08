# Compute a mask of all points in a volume that have a fiber running through them

from lazy_imports import np
import whitematteranalysis as wma

def get_mask_from_polydata(pd, xsz, ysz, zsz, spacing, origin):
  mask = np.zeros((xsz,ysz,zsz))



if __name__ == "__main__":
  fname = ''
  pd = wma.io.read_polydata(fname)

  get_mask_from_polydata(pd, xsz, ysz, zsz, spacing, origin)
