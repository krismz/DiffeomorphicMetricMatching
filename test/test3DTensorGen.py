# test basics of tensor field generation

from optparse import OptionParser
import pathlib
import sys
#from multiprocessing import Pool
import multiprocessing as mp
#import random
from lazy_imports import np

#from .. import data
#from .. import disp
#from ..data.io import WriteTensorNPArray
import data.gen
import disp
from data.io import WriteTensorNPArray, WriteScalarNPArray

#def testAnnulusParametric():
#  ann_p, ann_n = data.gen.make_annulus(1, 1 / 5.0, 10)
#  disp.vis.quiver_par_curv(ann_p, ann_n)

def testAnnulusTens3D(xsz, ysz, out_prefix, ratio=6.0):
  circ_rng = (-1.4,1.4)
  img, tens, seed, evals, xrg, yrg, zrg, zsz = data.gen.gen_3D_annulus(xsz, ysz, ratio, True, False, False, xrng = circ_rng, yrng = circ_rng, zrng=None, iso_scale=1.0)

  WriteTensorNPArray(tens, out_prefix + "tens.nhdr")
  WriteScalarNPArray(img, out_prefix + "mask.nhdr")
  WriteScalarNPArray(seed, out_prefix + "seed.nhdr")
  WriteTensorNPArray(evals, out_prefix + "evals.nhdr")

def testCubic1Tens3D(xsz, ysz, out_prefix, xrg=None, yrg=None, zrg=None, zero_padding_width=None):
  a3 = -4 #-4.0
  a2 = 4 #2.0
  a1 = 2 #10.0
  a0 = 0.0 # 0.0
  b3 = 8 #8.0
  b2 = -12 #-10.0
  b1 = 6 #2.0
  b0 = 0 #0
  ##a3 = -4.0
  ##a2 = 2.0
  ##a1 = 10.0
  ##a0 = 0.0
  ##b3 = 8.0
  ##b2 = -10.0
  ##b1 = 2.0
  ##b0 = 0
  #c1 = lambda t: data.gen.cubic(t, a3, a2, a1, a0)
  #dc1 = lambda t, dt: data.gen.d_cubic(t, a3, a2, a1, a0, dt)
  #c2 = lambda t: data.gen.cubic(t, b3, b2, b1, b0)
  #dc2 = lambda t, dt: data.gen.d_cubic(t, b3, b2, b1, b0, dt)
  ## TODO ok if xrg, yrg not isotropic?
  #(cubic_no_blur_img, cubic_no_blur_ten, cubic_no_blur_seed, cubic_no_blur_xrg, cubic_no_blur_yrg) = data.gen.gen_2D_tensor_image(xsz, ysz, 0, 1, 1000, c1, dc1, c2, dc2, 1/5.0, 15, 6.0, 0.05, 0.95,True,False,False, xrg, yrg, zero_padding_width=zero_padding_width)

  #WriteTensorNPArray(cubic_no_blur_ten, out_prefix + "tens.nhdr")
  #WriteScalarNPArray(cubic_no_blur_img, out_prefix + "mask.nhdr")
  #WriteScalarNPArray(cubic_no_blur_seed, out_prefix + "seed.nhdr")
  #return (cubic_no_blur_xrg, cubic_no_blur_yrg)

  return(testGenAndWriteCubic3D(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix, xrg, yrg, zrg, None, zero_padding_width))

def testCubic2Tens3D(xsz, ysz, out_prefix, xrg = None, yrg = None, zrg=None, zero_padding_width=None):
  a3_2 = -3 #-4 #-4.0
  a2_2 = 3.5 #4 #2.0
  a1_2 = 1.5 #2 #10.0
  a0_2 = 0.0 #0.0 # 0.0
  b3_2 = 6.5 #8 #8.0
  b2_2 = -11 #-12 #-10.0
  b1_2 = 6.5 #6 #2.0
  b0_2 = 0 #0 #0
  ##a3_2 = -3.0
  ##a2_2 = 1.5
  ##a1_2 = 9.5
  ##a0_2 = 0.0
  ##b3_2 = 6.5
  ##b2_2 = -9
  ##b1_2 = 2.5
  ##b0_2 = 0
  #c1_2 = lambda t: data.gen.cubic(t, a3_2, a2_2, a1_2, a0_2)
  #dc1_2 = lambda t, dt: data.gen.d_cubic(t, a3_2, a2_2, a1_2, a0_2, dt)
  #c2_2 = lambda t: data.gen.cubic(t, b3_2, b2_2, b1_2, b0_2)
  #dc2_2 = lambda t, dt: data.gen.d_cubic(t, b3_2, b2_2, b1_2, b0_2, dt)
  #(cubic_no_blur_img2, cubic_no_blur_ten2, cubic_no_blur_seed2, cubic_no_blur_xrg2, cubic_no_blur_yrg2) = data.gen.gen_2D_tensor_image(xsz, ysz, 0, 1, 1000, c1_2, dc1_2, c2_2, dc2_2, 1/5.0, 15, 6.0, 0.05, 0.95,True,False,False, xrg, yrg, zero_padding_width=zero_padding_width)

  #WriteTensorNPArray(cubic_no_blur_ten2, out_prefix + "tens.nhdr")
  #WriteScalarNPArray(cubic_no_blur_img2, out_prefix + "mask.nhdr")
  #WriteScalarNPArray(cubic_no_blur_seed2, out_prefix + "seed.nhdr")

  return(testGenAndWriteCubic3D(xsz, ysz, a0_2, a1_2, a2_2, a3_2, b0_2, b1_2, b2_2, b3_2, out_prefix, xrg, yrg, zrg, None, zero_padding_width))

def testGenAndWriteCubic3D(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix, xrg=None, yrg=None, zrg=None, fdist=None, ratio=None, template_stddevs=None, zero_padding_width=None, q=None, iso_scale=None):
  c1 = lambda t: data.gen.cubic(t, a3, a2, a1, a0)
  dc1 = lambda t, dt: data.gen.d_cubic(t, a3, a2, a1, a0, dt)
  c2 = lambda t: data.gen.cubic(t, b3, b2, b1, b0)
  dc2 = lambda t, dt: data.gen.d_cubic(t, b3, b2, b1, b0, dt)
  # TODO ok if xrg, yrg not isotropic?

  if fdist is None:
    fdist = 1/5.0
  if ratio is None:
    ratio = 6.0
    
  
  (cubic_no_blur_img, cubic_no_blur_ten, cubic_no_blur_seed, cubic_no_blur_rand_evals, cubic_no_blur_xrg, cubic_no_blur_yrg, cubic_no_blur_zrg, zsz) = data.gen.gen_3D_tensor_image(xsz, ysz, 0, 1, 1000, c1, dc1, c2, dc2, fdist, 15, ratio, 0.05, 0.95,True,False,False, xrg, yrg, zrg, template_stddevs=template_stddevs, zero_padding_width=zero_padding_width, iso_scale=iso_scale)

  if q is None:
    WriteTensorNPArray(cubic_no_blur_ten, out_prefix + "tens.nhdr")
    WriteScalarNPArray(cubic_no_blur_img, out_prefix + "mask.nhdr")
    WriteScalarNPArray(cubic_no_blur_seed, out_prefix + "seed.nhdr")
    WriteTensorNPArray(cubic_no_blur_rand_evals, out_prefix + "evals.nhdr")
  else:
    q.put((out_prefix, cubic_no_blur_ten, cubic_no_blur_img, cubic_no_blur_seed, cubic_no_blur_rand_evals))
  return (cubic_no_blur_xrg, cubic_no_blur_yrg, cubic_no_blur_zrg)

def write_to_file_listener(q):
    while True:
      m = q.get()
      if len(m) == 5:
        WriteTensorNPArray(m[1], m[0] + "tens.nhdr")
        WriteScalarNPArray(m[2], m[0] + "mask.nhdr")
        WriteScalarNPArray(m[3], m[0] + "seed.nhdr")
        WriteTensorNPArray(m[4], m[0] + "evals.nhdr")
      else:
        print('Received unexpected message on queue:', m)
        break
        

def testMultipleCubicTens3D(xsz, ysz, out_prefix, num_cubics=None, xrg=None, yrg=None, zrg=None, template_stddevs=None, zero_padding_width=None, iso_scale=None):
  a3 = -4 #-4.0
  a2 = 4 #2.0
  a1 = 2 #10.0
  a0 = 0.0 # 0.0
  b3 = 8 #8.0
  b2 = -12 #-10.0
  b1 = 6 #2.0
  b0 = 0 #0

  a3_2 = -3 #-4 #-4.0
  a2_2 = 3.5 #4 #2.0
  a1_2 = 1.5 #2 #10.0
  a0_2 = 0.0 #0.0 # 0.0
  b3_2 = 6.5 #8 #8.0
  b2_2 = -11 #-12 #-10.0
  b1_2 = 6.5 #6 #2.0
  b0_2 = 0 #0 #0

  if num_cubics is None:
    num_cubics = 2
  if num_cubics >= 1:
    xrg, yrg, zrg = testGenAndWriteCubic3D(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix + "cubic1_", xrg, yrg, zrg,
                                           template_stddevs=template_stddevs, zero_padding_width=zero_padding_width, iso_scale=iso_scale)
  else:
    return

  if num_cubics >= 2:
    testGenAndWriteCubic3D(xsz, ysz, a0_2, a1_2, a2_2, a3_2, b0_2, b1_2, b2_2, b3_2, out_prefix + "cubic2_", xrg, yrg, zrg,
                           template_stddevs=template_stddevs, zero_padding_width=zero_padding_width, iso_scale=iso_scale)
  else:
    return

  #random.seed(86) # for repeated random.uniform
  #rfs = [1, 0]
  #random.seed(867530)
  #for cc in range(3,num_cubics+1):
  #  rfs.append(random.uniform(0,1))
  rfs = [1, 0, 0.2, 0.4, 0.5, 0.6, 0.8]
  
  for cc in range(3,num_cubics+1):
    rf = rfs[cc-1]
    print("random percent",rf)
    a3_r = a3_2 + rf * (a3 - a3_2)
    a2_r = a2_2 + rf * (a2 - a2_2)
    a1_r = a1_2 + rf * (a1 - a1_2)
    a0_r = 0.0
    b3_r = b3_2 + rf * (b3 - b3_2)
    b2_r = b2_2 + rf * (b2 - b2_2)
    b1_r = b1_2 + rf * (b1 - b1_2)
    b0_r = 0.0
    #a3_r = random.uniform(a3, a3_2)
    #a2_r = random.uniform(a2_2, a2)
    #a1_r = random.uniform(a1_2, a1)
    #a0_r = 0.0
    #b3_r = random.uniform(b3_2, b3)
    #b2_r = random.uniform(b2, b2_2)
    #b1_r = random.uniform(b1, b1_2)
    #b0_r = 0.0
    testGenAndWriteCubic3D(xsz, ysz, a0_r, a1_r, a2_r, a3_r, b0_r, b1_r, b2_r, b3_r, out_prefix + f"cubic{cc}_" , xrg, yrg, zrg,
                           template_stddevs=template_stddevs, zero_padding_width=zero_padding_width, iso_scale=iso_scale)
  return(xrg, yrg, zrg)

#def testVaryRectangleTens3D(xsz, ysz, out_prefix):
#  tens = data.gen.gen_3D_rectangle_gradient_ratio(xsz, ysz, 1/6.0, rotation = 0, do_isotropic=True, zero_padding_width=0)
#  WriteTensorNPArray(tens, out_prefix + "rect_vary_tens.nhdr")

#def testIsoRectangleTens3D(xsz, ysz, out_prefix):
#  tens = data.gen.gen_3D_rectangle_constant_ratio(xsz, ysz, 1, rotation = 0, do_isotropic=True, zero_padding_width=0)
#  WriteTensorNPArray(tens, out_prefix + "rect_iso_tens.nhdr")

def testRandomCubicTens3D(xsz, ysz, out_prefix, num_cubics=None, xrg=None, yrg=None, zrg=None, no_shape_diff=None, fdists=None, ratios=None, group_stddevs=None, template_1_stddevs=None, template_2_stddevs=None, fdist_stddevs=None, ratio_stddevs=None, zero_padding_width=None, iso_scale=None):
  # Generate cubics from 2 different populations
  # fdists -- list of 2 widths of the cubic tube for each group
  # ratios -- list of 2 ratios of first to second eigenvector (third eigenvector == second eigenvector) for each group
  # group_stddevs -- list of std deviations around the cubic params (tube shape) for each group
  # template_1_stddevs -- std deviations for the template tensor for group 1
  # template_2_stddevs -- std deviations for the template tensor for group 2
  # fdist_stddevs -- list of std deviations around fdist for each group
  # ratio_stddevs -- list of std deviations around ratio for each group

  if no_shape_diff is None:
    no_shape_diff = False
  
  a3 = -4 #-4.0
  a2 = 4 #2.0
  a1 = 2 #10.0
  a0 = 0.0 # 0.0
  b3 = 8 #8.0
  b2 = -12 #-10.0
  b1 = 6 #2.0
  b0 = 0 #0

  if no_shape_diff:
    a3_2 = a3
    a2_2 = a2
    a1_2 = a1
    a0_2 = a0
    b3_2 = b3
    b2_2 = b2
    b1_2 = b1
    b0_2 = b0
  else:
    a3_2 = -3 #-4 #-4.0
    a2_2 = 3.5 #4 #2.0
    a1_2 = 1.5 #2 #10.0
    a0_2 = 0.0 #0.0 # 0.0
    b3_2 = 6.5 #8 #8.0
    b2_2 = -11 #-12 #-10.0
    b1_2 = 6.5 #6 #2.0
    b0_2 = 0 #0 #0


  log_1_file=out_prefix + 'group_1_cubics.txt'
  log_2_file=out_prefix + 'group_2_cubics.txt'

  log_1 = open(log_1_file, 'w')
  log_2 = open(log_2_file, 'w')
  
  if num_cubics is None:
    num_cubics = 2
  if num_cubics >= 1:
    #print('Gen 1st cubic, (xrg, yrg, zrg) = ', xrg,yrg,zrg)
    # First one must complete before running the rest so that we use a consistent xrg, yrg, zrg across all images
    xrg, yrg, zrg = testGenAndWriteCubic3D(xsz, ysz, a0, a1, a2, a3, b0, b1, b2, b3, out_prefix + "cubic1_novar_", xrg, yrg, zrg,
                                           fdist=fdists[0], ratio=ratios[0],
                                           template_stddevs=[0,0,0], zero_padding_width=zero_padding_width, iso_scale=iso_scale)
    #print('After gen 1st cubic, (xrg, yrg, zrg) = ', xrg,yrg,zrg)
    log_1.write('Prototype Cubic\n')
    log_1.write(f'{a3} {a2} {a1} {a0} {b3} {b2} {b1} {b0}\n')
    log_1.write(f'{fdists[0]} {ratios[0]}\n')
  else:
    return

  manager = mp.Manager()
  qq = manager.Queue()
  all_res = []
  
  with mp.Pool(min(20,2*num_cubics-1)) as p:
    
    p.apply_async(write_to_file_listener,(qq,))
    
    if num_cubics >= 2:

      res = p.apply_async(testGenAndWriteCubic3D,(xsz, ysz, a0_2, a1_2, a2_2, a3_2, b0_2, b1_2, b2_2, b3_2, out_prefix + "cubic2_novar_", xrg, yrg, zrg),
                       dict(fdist=fdists[1], ratio=ratios[1],
                                 template_stddevs=[0,0,0], zero_padding_width=zero_padding_width, q=qq, iso_scale=iso_scale))
      all_res.append(res)
      #print('After gen 2nd cubic, (xrg, yrg, zrg) = ', tmpx,tmpy,tmpz)
      log_2.write('Prototype Cubic\n')
      log_2.write(f'{a3_2} {a2_2} {a1_2} {a0_2} {b3_2} {b2_2} {b1_2} {b0_2}\n')
      log_2.write(f'{fdists[1]} {ratios[1]}\n')
    else:
      return

    # Now generate the ground truth mean of the two prototypes
    a3_m = (a3_2 + a3) / 2.0
    a2_m = (a2_2 + a2) / 2.0
    a1_m = (a1_2 + a1) / 2.0
    a0_m = (a0_2 + a0) / 2.0
    b3_m = (b3_2 + b3) / 2.0
    b2_m = (b2_2 + b2) / 2.0
    b1_m = (b1_2 + b1) / 2.0
    b0_m = (b0_2 + b0) / 2.0
    fdist_mean = (fdists[0] + fdists[1]) / 2.0
    ratio_mean = (ratios[0] + ratios[1]) / 2.0
    
    res = p.apply_async(testGenAndWriteCubic3D,(xsz, ysz, a0_m, a1_m, a2_m, a3_m, b0_m, b1_m, b2_m, b3_m, out_prefix + "cubic_1_2_mean_", xrg, yrg, zrg),
                        dict(fdist=fdist_mean, ratio=ratio_mean,
                             template_stddevs=[0,0,0], zero_padding_width=zero_padding_width, q=qq, iso_scale=iso_scale))
    all_res.append(res)
    #print('After gen 2nd cubic, (xrg, yrg, zrg) = ', tmpx,tmpy,tmpz)
    log_1.write('Mean Cubic\n')
    log_1.write(f'{a3_m} {a2_m} {a1_m} {a0_m} {b3_m} {b2_m} {b1_m} {b0_m}\n')
    log_1.write(f'{fdist_mean} {ratio_mean}\n')
    log_2.write('Mean Cubic\n')
    log_2.write(f'{a3_m} {a2_m} {a1_m} {a0_m} {b3_m} {b2_m} {b1_m} {b0_m}\n')
    log_2.write(f'{fdist_mean} {ratio_mean}\n')
    
 
    for cc in range(num_cubics):
      a3_r = np.random.normal(a3, group_stddevs[0])
      a2_r = np.random.normal(a2, group_stddevs[0])
      a1_r = np.random.normal(a1, group_stddevs[0])
      a0_r = 0.0
      b3_r = np.random.normal(b3, group_stddevs[0])
      b2_r = np.random.normal(b2, group_stddevs[0])
      b1_r = np.random.normal(b1, group_stddevs[0])
      b0_r = 0.0
      a3_2_r = np.random.normal(a3_2, group_stddevs[1])
      a2_2_r = np.random.normal(a2_2, group_stddevs[1])
      a1_2_r = np.random.normal(a1_2, group_stddevs[1])
      a0_2_r = 0.0
      b3_2_r = np.random.normal(b3_2, group_stddevs[1])
      b2_2_r = np.random.normal(b2_2, group_stddevs[1])
      b1_2_r = np.random.normal(b1_2, group_stddevs[1])
      b0_2_r = 0.0
      fdist_1 = np.random.normal(fdists[0], fdist_stddevs[0])
      ratio_1 = np.random.normal(ratios[0], ratio_stddevs[0])
      fdist_2 = np.random.normal(fdists[1], fdist_stddevs[1])
      ratio_2 = np.random.normal(ratios[1], ratio_stddevs[1])
      res = p.apply_async(testGenAndWriteCubic3D,(xsz, ysz, a0_r, a1_r, a2_r, a3_r, b0_r, b1_r, b2_r, b3_r, out_prefix + f"cubic1_{cc}_" , xrg, yrg, zrg),
                          dict(fdist=fdist_1, ratio=ratio_1,
                               template_stddevs=template_1_stddevs, zero_padding_width=zero_padding_width, q=qq, iso_scale=iso_scale))
      all_res.append(res)
      #print('After gen ', cc, 'th group 1 cubic, (xrg, yrg, zrg) = ', tmpx,tmpy,tmpz)
      log_1.write(f'{cc}th Cubic\n')
      log_1.write(f'{a3_r} {a2_r} {a1_r} {a0_r} {b3_r} {b2_r} {b1_r} {b0_r}\n')
      log_1.write(f'{fdist_1} {ratio_1}\n')
      
      res = p.apply_async(testGenAndWriteCubic3D,(xsz, ysz, a0_2_r, a1_2_r, a2_2_r, a3_2_r, b0_2_r, b1_2_r, b2_2_r, b3_2_r, out_prefix + f"cubic2_{cc}_" , xrg, yrg, zrg),
                          dict(fdist=fdist_2, ratio=ratio_2,
                               template_stddevs=template_2_stddevs, zero_padding_width=zero_padding_width, q=qq, iso_scale=iso_scale))
      all_res.append(res)
      #print('After gen ', cc, 'th group 2 cubic, (xrg, yrg, zrg) = ', tmpx,tmpy,tmpz)
      log_2.write(f'{cc}th Cubic\n')
      log_2.write(f'{a3_2_r} {a2_2_r} {a1_2_r} {a0_2_r} {b3_2_r} {b2_2_r} {b1_2_r} {b0_2_r}\n')
      log_2.write(f'{fdist_2} {ratio_2}\n')
    # end for each cubic
    
    log_1.close()
    log_2.close()
    
    for res in all_res:
      res.get()

    qq.put(('end'))  
    p.close()
    p.join()

  return(xrg, yrg, zrg)

#def testVaryRectangleTens3D(xsz, ysz, out_prefix):
#  tens = data.gen.gen_3D_rectangle_gradient_ratio(xsz, ysz, 1/6.0, rotation = 0, do_isotropic=True, zero_padding_width=0)
#  WriteTensorNPArray(tens, out_prefix + "rect_vary_tens.nhdr")

#def testIsoRectangleTens3D(xsz, ysz, out_prefix):
#  tens = data.gen.gen_3D_rectangle_constant_ratio(xsz, ysz, 1, rotation = 0, do_isotropic=True, zero_padding_width=0)
#  WriteTensorNPArray(tens, out_prefix + "rect_iso_tens.nhdr")


if __name__ == "__main__":
  usage = """
%prog [options]

generates an annulus and two different cubic tensor fields.  
"""
  optparser = OptionParser(usage=usage)
  optparser.add_option("-a", action="store_true",  dest="annulus_only",
                       help="Only generate annulus images.")
  optparser.add_option("-d", action="store_true",  dest="no_shape_diff",
                       help="No shape differences between groups.")
  optparser.add_option("-x", "--xsize", dest="xsz", default="100",
                       help="Size of output image in x direction.")
  optparser.add_option("-y", "--ysize", dest="ysz", default="100",
                       help="Size of output image in y direction.")
  optparser.add_option("-o", "--outdir", dest="outdir", default=".",
                       help="Directory to which resulting images will be written.")
  optparser.add_option("-n", "--numcubics", dest="numcubics", default="2",
                       help="Number of cubic functions to generate")
  optparser.add_option("-b", "--cubicstddev1", dest="cubicstddev1", default="0.1",
                       help="std deviation for cubic group 1")
  optparser.add_option("-c", "--cubicstddev2", dest="cubicstddev2", default="0.1",
                       help="std deviation for cubic group 2")
  optparser.add_option("-r", "--tensorratio1", dest="tensorratio1", default="6",
                       help="ratio between first and second eigenvectors of tensors for group 1")
  optparser.add_option("-s", "--tensorratio2", dest="tensorratio2", default="6",
                       help="ratio between first and second eigenvectors of tensors for group 2")
  optparser.add_option("-t", "--tensorstddev1", dest="tensorstddev1", default="0.1",
                       help="std deviation of tensor eigenvectors for group 1")
  optparser.add_option("-u", "--tensorstddev2", dest="tensorstddev2", default="0.1",
                       help="std deviation of tensor eigenvectors for group 2")
  optparser.add_option("-f", "--fdist1", dest="fdist1", default="0.2", # 1 / 5.0
                       help="width of cubic tube for group 1")
  optparser.add_option("-g", "--fdist2", dest="fdist2", default="0.2", # 1 / 5.0
                       help="width of cubic tube for group 2")
  optparser.add_option("-i", "--fdiststddev1", dest="fdiststddev1", default="0.01",
                       help="std deviation of cubic tube width for group 1")
  optparser.add_option("-j", "--fdiststddev2", dest="fdiststddev2", default="0.01",
                       help="std deviation of cubic tube width for group 2")
  optparser.add_option("-z", "--zeropadding", dest="zeropadding", default="20",
                       help="voxels of zero padding on each side of volume")
  
  (options, args) = optparser.parse_args()

  pathlib.Path(options.outdir).mkdir(exist_ok=True)
  #print('WARNING!! Skipping annulus')
  testAnnulusTens3D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_annulus_3D_1_",ratio=6.0)
  testAnnulusTens3D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_annulus_3D_2_",ratio=5.0)
  if options.annulus_only:
    sys.exit(1)
  cubic_xrg = [-0.4, 2.6]
  cubic_yrg =[-0.4, 2.6]
  cubic_xrg = [-1.25, 8.4]
  cubic_yrg =[-1.25, 8.4]
  cubic_xrg = None
  cubic_yrg = None
  cubic_zrg = None

  testRandomCubicTens3D(int(options.xsz), int(options.ysz), options.outdir + "/metpy_3D_", int(options.numcubics), cubic_xrg, cubic_yrg, cubic_zrg, no_shape_diff = options.no_shape_diff,
                        fdists=[float(options.fdist1),float(options.fdist2)], ratios=[float(options.tensorratio1),float(options.tensorratio2)],
                        group_stddevs=[float(options.cubicstddev1),float(options.cubicstddev2)], 
                        template_1_stddevs=[float(options.tensorstddev1),float(options.tensorstddev1),float(options.tensorstddev1)],
                        template_2_stddevs=[float(options.tensorstddev2),float(options.tensorstddev2),float(options.tensorstddev2)],
                        fdist_stddevs=[float(options.fdiststddev1),float(options.fdiststddev2)], ratio_stddevs=[float(options.tensorstddev1),float(options.tensorstddev2)],
                        zero_padding_width=int(options.zeropadding), iso_scale=1.0)

  #testVaryRectangleTens3D(10, 5, options.outdir + "/metpy_3D_")
  #testIsoRectangleTens3D(10, 5, options.outdir + "/metpy_3D_")
