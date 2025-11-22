"""A whole bunch of math functions essential to build the project."""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Import & process parameters 

import numpy as np
from config import *
# from debuggerFc import *
import time

perpDotMax = np.abs(np.dot(np.array([1,0]), np.array([np.sin(perpendicular_range*np.pi/180.0), np.cos(perpendicular_range*np.pi/180.0)])))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## System utility

def rl(arr): # fck
    """range(len(arr))"""
    return range(len(arr))

def tic(): # from prof
  """Get time"""
  return time.time()

def toc(tstart, name="", prt=True): # from prof
  """Print/get time used"""
  tend = time.time()
  if prt:
    print('%s took: %s sec.\n' % (name,(tend - tstart)))
  return tend-tstart

def isTensPercent(i, max): # fck
  """Returns if i is x*10% of max, where x is an integer"""
  if max//10 == 0: 
      return True
  return i % np.floor(max/10) == 0

def listConcateWithCond(lst, condArr, cond): 
    """np.concatenate(lst[condArr==cond], axis=0) except for list 'cause that doesn't work for list. Assumes len(condArr) == len(lst)"""
    assert len(condArr) == len(lst)
    rtn = []
    for i in range(len(lst)): 
        if condArr[i] == cond: 
            for j in range(len(lst[i])):
                rtn.append(lst[i][j])
    return np.array(rtn)

def getFileName(file_dir): # ck
    """Extract file name from file dir. Is literally getting the string between the last / and ."""
    slashIdx = 0
    dotIdx = 0
    for i in range(len(file_dir)-1,-1,-1): 
        if file_dir[i] == '.': 
            dotIdx = i
            break
    for i in range(dotIdx-1,-1,-1): 
        if file_dir[i] == '/': 
            slashIdx = i
            break
    return file_dir[slashIdx+1:dotIdx]

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Basic math

def mirror(num, axle): # fck
    """'mirrors' num agains axle. e.g. mirror(1.5,1) -> 0.5, mirror(-0.2,2) -> 4.2, mirror(5,-2) -> -9
    
    Does support nd array with broadcasting"""
    return 2*axle - num

def R2Euler(R): # ck
    roll = np.arctan2(R[...,2,1], R[...,2,2])
    pitch = np.arcsin(-R[...,2,0])
    yaw = np.arctan2(R[...,1,0], R[...,0,0])
    return firstDimToLast(np.array([roll, pitch, yaw]))

def R2EulerNumHelper(euler,R): # ck
    """Wrapper to numerically solve euler angle"""
    Rpred = Euler2R(euler)
    return np.sum((R-Rpred)**2,axis=(-2,-1))

def R2EulerNum(R, iniGuess=np.array([])): # ck
    """Numerically solve euler angle 'cause the stupid euler lock
    You SHOULD use the analytical solver whenever possible."""
    # LOL NUMERICAL SOLVER LETS GO LOLLLL
    if len(iniGuess) == 0:
        size = R.shape
        return solvePGDMulti(np.random.random_sample(size[:-1]),R,R2EulerNumHelper,randMax=2*np.pi,showStep=False,tol=1e-6)
    return solvePGDMulti(iniGuess,R,R2EulerNumHelper,randMax=2*np.pi,showStep=False,tol=1e-6)

def Euler2R(euler): # ck
    c = np.cos(euler)
    s = np.sin(euler)
    Rm = firstDimToLast(np.array([[c[...,1]*c[...,2], s[...,0]*s[...,1]*c[...,2]-c[...,0]*s[...,2], c[...,0]*s[...,1]*c[...,2]+s[...,0]*s[...,2]], 
            [c[...,1]*s[...,2], s[...,0]*s[...,1]*s[...,2]+c[...,0]*c[...,2], c[...,0]*s[...,1]*s[...,2]-s[...,0]*c[...,2]], 
            [-s[...,1], s[...,0]*c[...,1], c[...,0]*c[...,1]]]))
    return Rm 

def vecdir2thetaphi(vec): # ck
    """[x,y,z] -> [phi,theta]
    [x,y,z] are automatically unit-rized. When theta=0, default to phi=0."""
    vv = vec / dimUp(np.linalg.norm(vec,axis=-1))
    vz = np.zeros_like(vv)
    vz[...,0] = vv[...,0]
    vz[...,1] = vv[...,1]
    vz = vz / dimUp(np.linalg.norm(vz,axis=-1))
    rtn = np.array([np.arccos(np.dot(vz,[1,0,0])),np.arccos(np.dot(vv,[0,0,1]))]).T
    return np.nan_to_num(rtn,copy=False)

def thetaphi2vecdir(ang): # fck
    """[phi,theta] -> [x,y,z]_unit"""
    s = np.sin(ang)
    c = np.cos(ang)
    return np.array([s[...,1]*c[...,0],s[...,1]*s[...,0],c[...,1]]).T

def batchRotate(R,vs): # ck
    """Rotates all vs [nxd] at once by R [3x3]"""
    return np.einsum('ij,kj->ki',R,vs)

def batchDot(vecss, vecs): # fck
    """Batch np.dot(vecss,each_of_vecs). 
    vecss in nxmxg
    vecs in vxg
    result in nxvxm"""
    return np.einsum('ijk,uk->iuj',vecss,vecs)

def circularAdd(a,b,max,min=0.0): # fck
    """a+b but circles in [min,max). 
    
    e.g. circleAdd(1,2,4) -> 3, circleAdd(1,2,3) -> 0, circleAdd(1,2,2) -> 1, circleAdd(1,2,5,min=2) -> 3, circleAdd(1,2,0,min=-5) -> -2"""
    raw = a+b-min
    mm = max-min
    raw = raw % mm + min
    while raw < min: 
        raw += mm
    return raw

def circularAddInt(a,b,max,min=0): # fck
    """a+b but circles in [min,max). Integer version. 
    
    e.g. circleAdd(1,2,4) -> 3, circleAdd(1,2,3) -> 0, circleAdd(1,2,2) -> 1, circleAdd(1,2,5,min=2) -> 3, circleAdd(1,2,0,min=-5) -> -2"""
    raw = a+b-min
    mm = max-min
    raw = raw % mm + min
    while raw < min: 
        raw += mm
    return raw

def circularMinusAbs(a,b,max): # ck
    """abs(a-b) but returns the minimum value that circles in [0,max). 
    
    e.g. circularMinusAbs(1,2,7) -> 1, circularMinusAbs(1,6,7) -> 2"""
    dir = np.abs(a-b)
    indir = np.min([max-b+a,max-a+b])
    return np.min([dir,indir])

def inBetween(a, bounds): # fck
    """Returns a < bounds[0] and a > bounds[1] or a > bounds[0] and a < bounds[1]"""
    return (a < bounds[0] and a > bounds[1]) or (a > bounds[0] and a < bounds[1])

def multiInBetween(aaa, bounds): # fck
    """inBetween(aaa[0], bounds[0]) and inBetween(aaa[1], bounds[1]) and inBetween(aaa[2], bounds[2]) and..."""
    if len(aaa) != len(bounds): 
        print('multiInBetween received non-matching lengthes! aaa is', len(aaa), 'long but bounds are', len(bounds), 'long.')
        return False
    for i in range(len(aaa)): 
        if not inBetween(aaa[i],bounds): 
            return False
    return True

def otherone(i): # fck
    """Helper function. Returns i+1 if i%2 == 0, i-1 if i%2 != 0"""
    if i%2 == 0: 
        return i+1
    return i-1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Array/list operations

def ea1(arr): 
    """np.expand_dims(arr,axis=1)"""
    return np.expand_dims(arr,axis=1)

def blurZero(arr, tol=coincident_tol): # ck
    """Converges elements in arr below tol to 0. Used to deal with float point issues. Works both by return or by calling. 

    ***MUTATES THE INPUT ARRAY***
    """
    arr[np.abs(arr)<tol] = 0
    return arr

def arrContainNorm(arr, ele): # ck
    """listContain but uses np.linalg.norm against the last axis, for numpy arraies only. Returns all indexes that equals ele. If none, returns an empty array
    
    Is faster than arrContain."""
    if len(arr) <= 0: 
        return np.array([])
    diff = blurZero(np.linalg.norm(arr-ele, axis=-1),coincident_tol)
    return np.argwhere(diff == 0)

def listContain(lst, ele): # fck
    """lst.contains(ele), dummy O(n). Returns the first equal index if True, -1 if False"""
    for i in range(len(lst)): 
        if lst[i] == ele: 
            return i
    return -1

def swapEle(arr, index1=0, index2=1): 
    """Swaps elements on index1 and index2 of the last axis of arr. e.g. swapEle([[...[x,y,z]...]],0,1) -> [[...[y,x,z]...]]"""
    assert index1 < arr.shape[-1] and index2 < arr.shape[-1]
    abb = np.zeros_like(arr)
    for i in range(abb.shape[-1]): 
        if i == index1 or i == index2: 
            continue
        abb[...,i] = arr[...,i]
    abb[...,index1] = arr[...,index2]
    abb[...,index2] = arr[...,index1]
    return abb

def shiftLeft(arr): # ck
    """Shifts (circular) the last dimension of arr to the left, so e.g. [[...[x,y,z]...]] -> [[...[y,z,x]...]]
    arr in np.array
    Not better than numpy.roll"""
    return np.roll(arr,-1,-1)

def shiftRight(arr): # ck
    """Shifts (circular) the last dimension of arr to the right, so e.g. [[...[x,y,z]...]] -> [[...[z,x,y]...]]
    arr in np.array"""
    return np.roll(arr,1,-1)

def dimUp(arr): # fck
    """[axbx...xc] -> [axbx...xcx1]"""
    return np.expand_dims(arr,axis=-1)

def sharedElements(arr1, arr2, index=False): # ck
    """Return elements that are both in arr1 and arr2. Assumes arr1 in [mx[element]] and arr2 in [nx[element]]

    If index = True, return indexes of equal elements instead. Indexes in np.array([[index_in_arr1, index_in_arr2]_pair1, [index_in_arr1, index_in_arr2]_pair2,...])"""
    if len(arr1) == 0 or len(arr2) == 0:
        return np.array([])
    eAxis = tuple(np.arange(1,arr1.ndim)+1)
    a2 = np.expand_dims(arr2,axis=1)
    if len(eAxis) < 1: 
        diff = arr1 - a2
    else:
        diff = np.linalg.norm(arr1 - a2, axis=eAxis)
    same = np.argwhere(diff==0)
    if len(same) < 1: 
        return np.array([])
    if index: 
        return np.flip(same,axis=1)
    return arr1[same[:,1]]

def flatArr(arr): # ck
    if len(arr.shape) <= 1: 
        return arr
    return flatArr(np.concatenate(arr,axis=0))

def transpose2D(arr, dim1=0, dim2=1): # fck
    """np.transpose just in a better way for the most common usage"""
    axs = np.arange(arr.ndim)
    axs[dim1] = dim2
    axs[dim2] = dim1
    return np.transpose(arr,axes=axs)

def firstDimToLast(arr): # fck
    """arr.shape = [axnxmx...] -> [nxmx...xa]"""
    axs = shiftRight(np.arange(arr.ndim))
    return np.transpose(arr,axes=axs)

def roughEqual(lst, tol): # ck
    """avg(lst) +- tol == lst[0] == lst[1] == lst[2] == ...
    if lst[0] has more than 1 element, it is checked against the first axis ONLY (e.g. lst[0,0]==lst[1,0]==lst[2,0], lst[0,1]==lst[1,1]==lst[2,1], ...)"""
    l = np.array(lst)
    avg = np.average(l,axis=0)
    mx = avg + tol
    mn = avg - tol
    dx = flatArr(l - mx)
    for i in range(len(dx)): 
        if dx[i] > 0: 
            return False
    dn = flatArr(l - mn)
    for i in range(len(dn)):
        if dn[i] < 0: 
            return False
    return True

def removeIdentical(arr): # ck
    """'Removes' identical elements from an array. Returns the no-overlapping array. Computes along the first axis only."""
    diff = np.abs(blurZero(arr - ea1(arr)))
    while diff.ndim > 2: 
        diff = np.sum(diff,axis=-1)
    rtn = []
    repeats = np.zeros(len(arr))
    for i in range(len(arr)): 
        if repeats[i] == 1: 
            continue
        if len(diff[i][diff[i]==0]) > 1: 
            rtn.append(arr[i])
            for j in range(len(diff[i])): 
                if diff[i][j] == 0: 
                    repeats[j] = 1
            continue
        rtn.append(arr[i])
    return np.array(rtn)

def dupArr(arr): # ck
    """na.([a,b,c,...]) -> na.([a,a,b,b,c,c,...])"""
    return np.concatenate(np.stack((arr,arr),axis=1),axis=0)

def toList1D(arr, reverse=False): # fck
    """arr [nxaxbx...] -> list n x na. of [axbx...]
    If reverse, also reverses the order of elements."""
    rtn = []
    l = len(arr)
    for i in range(l): 
        if reverse: 
            rtn.append(arr[l-i-1])
        else:
            rtn.append(arr[i])
    return rtn.copy()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Vector operations (3D/nD)

def getANorm(vecs): # fck
    """Returns a vector normal to vec for each vec in vecs. Accepts vecs in [nxd], returns normal vectors in [nxd]. Is for 3D and returns a 'random' norm vector."""
    x = np.array([1,0,0])
    nv = np.cross(vecs, x)
    nl = np.linalg.norm(nv,axis=-1)
    if len(nl[nl==0]) <= 0: 
        return nv
    y = np.array([0,1,0])
    nv[nl==0] = np.cross(vecs[nl==0],y)
    return toUnit(nv)

def toUnit(vecs): # fck
    """vecs [axbx...xd] -> unit vectors [axbx...xd]. Unit-izes along the last axis. Works for 1d arrays too"""
    if len(vecs.shape) <= 1: 
        l = np.max(np.array([np.linalg.norm(vecs),1e-12]))
        return vecs / l
    ls = np.linalg.norm(vecs,axis=-1)
    ls[ls==0] = 1
    return vecs / dimUp(ls)

def copy2quads(dirs): # fck
    """Helper function. Copies angs to all 8 quads each [90,90]deg in theta_phi. each dir in dirs in xyz, in range theta,phi ∈ [0,90]degs"""
    out = np.zeros([8,len(dirs),len(dirs[0])])
    out[0] = dirs
    out[1] = -dirs
    out[2,:,:2] = dirs[...,:2]
    out[2,:,2] = -dirs[...,2]
    out[3,:,:2] = -dirs[...,:2]
    out[3,:,2] = dirs[...,2]
    out = np.concatenate(np.array(out),axis=0)
    rot90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    out[4*len(dirs):] = batchRotate(rot90,out[:4*len(dirs)])
    return np.array(out)

def eqSphereSample(ptPer90Deg): # ck
    """Generate arc-length-wise equally distant points around a sphere. 
    Input specifies how many points are on a 90-degree arc (so output has ~16/pi*ptPer90Deg^2)
    Returns as vector directions (e.g. np.array([[1,0,0],[0,1,0],...]))
    """
    if ptPer90Deg <= 0: 
        print('eqSphereSample() received illegal argument! Received points per 90 degree is:',ptPer90Deg)
        return np.array([])
    if ptPer90Deg == 1: # return "X"
        return np.concatenate(copy2quads(np.array([[1,1,1]])/np.sqrt(3)),axis=0)
    phis = np.linspace(0,np.pi/2,ptPer90Deg) # Create evenly-distributed points along a 90 deg arc
    arcLen = phis[1] - phis[0]
    out = [[0,phis[0]]]
    for i in range(1,len(phis)): # For each point on that arc, generate points with equal (with rounding) arc length on x-y plane
        num = int(np.floor((np.sin(phis[i])/2*np.pi+0.5*arcLen)/arcLen)) + 1
        spots = np.linspace(0,np.pi/2,num-1,endpoint=False)
        for s in spots: 
            out.append([s,phis[i]])
    out = np.array(out)
    out = thetaphi2vecdir(out)
    return copy2quads(out)

def selfBatchCross(vecss): # ck
    """Takes nxmx3 np.array and output nxmx3 np.array with each output[i,j,:] = np.cross(vecss[i,j,:],vecss[i,j+1,:]) (wraps around)"""
    b388i3335d7 = np.zeros_like(vecss)
    b388i3335d7[:,-1] = np.cross(vecss[:,-1],vecss[:,0])
    b388i3335d7[:,:-1] = np.cross(vecss[:,:-1],vecss[:,1:])
    return b388i3335d7

def selfBatchDot(vecss): # ck
    """Takes nxmxd np.array and output nxm np.array with each output[i,j] = np.dot(vecss[i,j,:],vecss[i,j+1,:]) (wraps around)"""
    bv1mNdqY8Eg5 = np.zeros([vecss.shape[0],vecss.shape[1]])
    bv1mNdqY8Eg5[:,-1] = np.einsum('nd,nd->n',vecss[:,-1],vecss[:,0])
    bv1mNdqY8Eg5[:,:-1] = np.einsum('nmd,nmd->nm',vecss[:,:-1],vecss[:,1:])
    return bv1mNdqY8Eg5

def batchIntersect3DCore2(pt, dir, ori, e1, e2): # ck
    """Core computation part of batchCoincident3D. Yes matrix would seem a lot more elegent, but that doesn't support MD and is way slower than hand-solved O(1).
    
    pt in [mxd] (support [d]), dir in [mxd]_unit_vector, ori in [nxd], e1 and e2 in [nxd]_unit_vector
    
    Returns an intersection result [nxm] in unit of times dir and a valid mask [nxm] that is 0 where a zero-division is encountered in calculation. Wherever had a zero division during calculation will have an invalid, basically made-up, result."""
    # 'Smarter' hand-solved version
    invalidMsk = np.zeros([len(ori),len(dir)])
    p = pt
    px = p[...,0]
    py = p[...,1]
    pz = p[...,2]
    o = ea1(ori)
    ox = o[...,0]
    oy = o[...,1]
    oz = o[...,2]
    v1 = ea1(e1)
    v1x = v1[...,0]
    v1y = v1[...,1]
    v1z = v1[...,2]
    v2 = ea1(e2)
    v2x = v2[...,0]
    v2y = v2[...,1]
    v2z = v2[...,2]
    v3 = dir
    v3x = v3[...,0]
    v3y = v3[...,1]
    v3z = v3[...,2]
    invalidMsk[:,v3x == 0] = coincident_tol
    v3xi = v3x+invalidMsk
    bb = v2y - v2x/v3xi * v3y
    invalidMsk[bb == 0] = coincident_tol
    bc = (py - oy + v2y/v3xi*(ox - px)) / (bb+invalidMsk)
    be = (v1y - v1x/v3xi*v3y) / (bb+invalidMsk)
    ab = v1z - v2z*be - v2z/v3xi*(v1x-v2x*be)
    invalidMsk[ab == 0] = coincident_tol
    a = (pz - oz + v3z/v3xi * (ox+bc*v2x) - bc*v2z) / (ab+invalidMsk)
    b = bc - be*a
    c = (ox + a*v1x + b*v2x - px) / v3xi
    return c, 1-np.sign(invalidMsk)

def batchIntersect3DCore(pt, dir, ori, e1, e2): # ck
    """Core computation part of batchCoincident3D. Yes matrix would seem a lot more elegent, but that is way slower than hand-solved O(1).
    
    pt in [mxd] (support [d]), dir in [mxd]_unit_vector, ori in [nxd], e1 and e2 in [nxd]_unit_vector
    
    Returns an intersection result [nxm] in unit of times dir and a valid mask [nxm] that is 0 where a zero-division is encountered in calculation. Wherever had a zero division during calculation will have an invalid, basically made-up, result."""
    # Literally Guassian reduction
    invalidMsk = np.zeros([len(ori),len(dir)])
    d = pt - ea1(ori)
    dx = d[...,0]
    dy = d[...,1]
    dz = d[...,2]
    v1 = ea1(e1)
    v1x = v1[...,0]
    v1y = v1[...,1]
    v1z = v1[...,2]
    v2 = ea1(e2)
    v2x = v2[...,0]
    v2y = v2[...,1]
    v2z = v2[...,2]
    v3 = dir
    v3x = v3[...,0]
    v3y = v3[...,1]
    v3z = v3[...,2]
    invalidMsk[np.abs(np.squeeze(v1x)) < 1e-8] = 1e-8
    v1xi = v1x + invalidMsk

    v2yi = v2y - v1y/v1xi*v2x
    v3yi = v3y - v1y/v1xi*v3x
    dyi = dy - v1y/v1xi*dx
    v2zi = v2z - v1z/v1xi*v2x
    v3zi = v3z - v1z/v1xi*v3x
    dzi = dz - v1z/v1xi*dx
    invalidMsk[np.abs(np.squeeze(v2yi)) < 1e-8] = 1e-8
    v2yi += invalidMsk

    v3zi = v3zi - v2zi/v2yi*v3yi
    dzi = dzi - v2zi/v2yi*dyi
    invalidMsk[np.abs(v3zi) < 1e-8] = 1e-8
    v3zi += invalidMsk

    return -dzi/v3zi, 1-np.sign(invalidMsk)

def batchIntersect3d(pt, dirs, trigs): # ck
    """Batch computes intersection points of pt to each direction in dirs for each triangle in trigs. 
    
    Input dirs in [mxd], trigs in [nx3xd]. Returns intersection in [nxm] in unit of times dirs, and validMsk [nxm] that is 0 where no intersection is found."""
    ori = trigs[:,0]
    e1 = toUnit(trigs[:,1] - trigs[:,0])
    e2 = toUnit(trigs[:,2] - trigs[:,0])
    c1, m1 = batchIntersect3DCore(pt, dirs, ori, e1, e2)
    rst = c1
    msk = m1
    if fast_intersect: 
        msk = np.ones_like(msk)
        return rst, msk

    pt1 = shiftRight(pt)
    dirs1 = shiftRight(dirs)
    ori1 = shiftRight(ori)
    e11 = shiftRight(e1)
    e21 = shiftRight(e2)
    c2, m2 = batchIntersect3DCore(pt1, dirs1, ori1, e11, e21)
    rst[m2 == 1] = c2[m2 == 1]
    msk += m2

    pt1 = shiftRight(pt1)
    dirs1 = shiftRight(dirs1)
    ori1 = shiftRight(ori1)
    e11 = shiftRight(e11)
    e21 = shiftRight(e21)
    c3, m3 = batchIntersect3DCore(pt1, dirs1, ori1, e11, e21)
    rst[m3 == 1] = c3[m3 == 1]
    msk += m3

    pt1 = swapEle(pt,0,1)
    dirs1 = swapEle(dirs,0,1)
    ori1 = swapEle(ori,0,1)
    e11 = swapEle(e1,0,1)
    e21 = swapEle(e2,0,1)
    c4, m4 = batchIntersect3DCore(pt1, dirs1, ori1, e11, e21)
    rst[m4 == 1] = c4[m4 == 1]
    msk += m4

    pt1 = swapEle(pt,0,2)
    dirs1 = swapEle(dirs,0,2)
    ori1 = swapEle(ori,0,2)
    e11 = swapEle(e1,0,2)
    e21 = swapEle(e2,0,2)
    c5, m5 = batchIntersect3DCore(pt1, dirs1, ori1, e11, e21)
    rst[m5 == 1] = c5[m5 == 1]
    msk += m5

    pt1 = swapEle(pt,2,1)
    dirs1 = swapEle(dirs,2,1)
    ori1 = swapEle(ori,2,1)
    e11 = swapEle(e1,2,1)
    e21 = swapEle(e2,2,1)
    c6, m6 = batchIntersect3DCore(pt1, dirs1, ori1, e11, e21)
    rst[m6 == 1] = c6[m6 == 1]
    msk += m6

    return rst, np.sign(msk)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Vector operations (2D/1D)

def inPlaneNorm(dir): # ck
    """Returns the in-plane normal direction to dir (in 2D coords). If dir is MD [axbx...x2], returns normal to each of the last dim so return will be in [axbx...x2]
    Returning directions are normalized"""
    assert dir.shape[-1] == 2
    rtn = np.ones_like(dir)*1.0
    slope = dir[...,1] / (blurZero(dir[...,0],2e-12) + 1e-12)
    norm = -1.0/(slope + 1e-12)
    rtn[...,1] = norm
    rtn /= dimUp(np.linalg.norm(rtn,axis=-1))
    return blurZero(rtn,2e-12)

def eqCircularSample(ptCt, maxAngle=2*np.pi, minAngle=0.0): # ck
    """Returns ptCt equally-spaced (angle-wise) sampling vectors. Optional [minAngle, maxAngle) argment specifies the angular interval where the samples can lie in
    
    Returns direction unit vectors [mx2]"""
    samples = np.arange(minAngle,maxAngle,(maxAngle-minAngle)/ptCt)
    return ang2Vec(samples)

def ang2Vec(angs): # fck
    """Convert 2D angles [m] to unit voctors [mx2]. If input is scaler, output a unit vector [2]"""
    y = np.sin(angs)
    x = np.cos(angs)
    if isinstance(angs, float) or isinstance(angs, int): 
        return np.array([x,y])
    return np.stack([x,y],axis=-1)

def vec2Ang(vecs): 
    """Convert 2D vectors [mx2] to angles [m]. If input is 1D, returns a scaler."""
    return np.arctan2(vecs[...,1],vecs[...,0])

def batchCross(vecss, vecs): # ck
    """A faster way to realize: (vecss in np.array([[[x,y,z]_1,...],[...],...]) [axbxd], vecs in np.array([[x,y,z]_1,[x,y,z]_2,...])) [cxd]

    s1 = vecss.shape
    s2 = vecs.shape
    out = np.zeros([s1[0],s2[0],s1[1],s1[2]])
    for i in range(len(vecss)): 
        for j in range(len(vecss[0])): 
            for k in range(len(vecs)): 
                out[i,j,k,:] = np.cross(vecss[i,j],vecs[k])
    return out (in [axbxcxd])

    That instead compresses the triple-for-loop into MD array operations. 
    """
    spe = vecss.shape
    vecsl = shiftLeft(vecs)
    vecsr = shiftRight(vecs)
    vss = np.concatenate(vecss,axis=0)
    vss = np.expand_dims(vss,axis=1)
    b3883 = shiftLeft(vss * vecsl) # y1 * z2 -> x
    d3357 = shiftRight(vss * vecsr) # y1 * x2 -> z
    return np.transpose(np.reshape(b3883 - d3357,[spe[0],spe[1],-1,spe[2]]),axes=[0,2,1,3])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Edge/point operations

def batchIntersect(starts, dirs, edges): # ck
    """Compute intersection points of start+dir to all edge in edges for all start in starts and all dir in dirs (assuming each start corresponds to 1 dir in dirs)

    starts in [n x points], dirs in [n x unit_vectors], edges in [m x 2 x vortices].

    If not using the MD feature, you can set m or n to be 1 (i.e. np.array([[stuff]]))
    For those intersecting out of the edge or are parallel, the distance will be -1. (The intersection will be a computed false point)

    Natively developed for 2D cases. May not work for 3D very well. 
    Look scary but it's merely a bunch of copies and a tiny bit of math. 

    Returns a list of intersection points and a list of distances (in times dir). ([n x m x dim]_intersections, [n x m]_distances).  """
    assert len(dirs) == len(starts) and starts.shape[-1] == dirs.shape[-1] == edges.shape[-1]
    se = edges.shape
    ss = starts.shape
    sm = (ss[0],se[0])
    e1xr = edges[...,0,0]
    e1yr = edges[...,0,1]
    e2xr = edges[...,1,0]
    e2yr = edges[...,1,1]
    sxr = starts[...,0]
    syr = starts[...,1]
    dxr = dirs[...,0]
    dyr = dirs[...,1]
    edxr = e2xr - e1xr
    edyr = e2yr - e1yr
    
    e1x = np.zeros_like(e1xr)
    e1y = np.zeros_like(e1yr)
    e2x = np.zeros_like(e2xr)
    e2y = np.zeros_like(e2yr)
    sx = np.zeros(sm)
    sy = np.zeros(sm)
    dx = np.zeros(sm)
    dy = np.zeros(sm)
    edx = np.zeros_like(edxr)
    edy = np.zeros_like(edyr)
    for i in range(len(edxr)): 
        if np.abs(edxr[i]) <= 1e-6: # If too vertical, swap to y-based route
            e1x[i] = e1yr[i]
            e1y[i] = e1xr[i]
            e2x[i] = e2yr[i]
            e2y[i] = e2xr[i]
            sx[:,i] = syr
            sy[:,i] = sxr
            dx[:,i] = dyr
            dy[:,i] = dxr
            edx[i] = edyr[i]
            edy[i] = edxr[i]
        else: 
            e1x[i] = e1xr[i]
            e1y[i] = e1yr[i]
            e2x[i] = e2xr[i]
            e2y[i] = e2yr[i]
            sx[:,i] = sxr
            sy[:,i] = syr
            dx[:,i] = dxr
            dy[:,i] = dyr
            edx[i] = edxr[i]
            edy[i] = edyr[i]
    assert len(edx[edx==0]) == 0
    edx[edx==0] = 1
    b0 = (sx-e1x)/edx * edy
    b0 = e1y + b0 - sy
    b1 = dy - edy/edx * dx
    n = -1 * np.ones_like(b0)
    nz = np.argwhere(b1 != 0).T
    n[nz[0],nz[1]] = b0[nz[0],nz[1]]/b1[nz[0],nz[1]]
    blurZero(n)
    its = ea1(starts) + dimUp(n)*ea1(dirs) # nxmxd
    linesL = np.linalg.norm(edges[:,1]-edges[:,0], axis=-1) # m
    y38i833n = np.abs(np.einsum('nmd,md->nm',its - edges[:,0], toUnit(edges[:,1]-edges[:,0]))) 
    h35e7 = np.abs(np.einsum('nmd,md->nm',its - edges[:,1], toUnit(edges[:,0]-edges[:,1])))
    n[linesL+coincident_tol < y38i833n+h35e7] = -1 # It'll broadcast
    n[n<0] = -1
    return its, n

def areOverlapping(pt1, pt2): # ck
    """Returns if two points are overlapping (below coincident_tol)"""
    return np.linalg.norm(pt1-pt2) < coincident_tol

def concateEdges(egs): # ck
    """Helper function. Concatinates all edges in egs. Does NOT check if they are colinear. egs [mx2xd] -> single edge [2xd] covering all edges in egs, assuming all colinear"""
    start = egs[0,0]
    dir = egs[0,1] - start
    nd0 = np.linalg.norm(dir,axis=-1)
    dir /= nd0
    rela = np.reshape(blurZero(np.dot(egs - start,dir)),(-1)) # mx2
    mini = np.argmin(rela)
    maxi = np.argmax(rela)
    return np.array([egs[mini//2,mini%2], egs[maxi//2,maxi%2]])

def mergeEdgesHelper(esout, nout, es, n, pt, coins): # ck
    """Helper function."""
    ck = np.sum(coins,axis=-1)
    for i in range(len(es)): 
        if ck[i] == 0:
            esout.append(es[i])
            nout.append(n[i])
            continue
        pti = pt[coins[i]==1]
        relPti = np.linalg.norm(pti - es[i,0],axis=-1)
        pti = pti[np.argsort(relPti)]
        esout.append(np.array([es[i,0],pti[0]]))
        nout.append(n[i])
        for j in range(len(pti)-1): 
            esout.append(np.array([pti[j],pti[j+1]]))
            nout.append(n[i])
        esout.append(np.array([pti[-1],es[i,1]]))
        nout.append(n[i])
    return esout, nout

def mergeEdges(es1, es2, n1, n2): # ck
    """Merges two lists of edges into one. Cut edges into segments from where at least two edges coincident, if is not cut already.
    ASSUMES no overlapping edges. 
    
    es1 in [mx2xd], es2 in [nx2xd], returns merged list of edges [(>=m+n)x2xd]
    Merges norms alongside edges ([mxd] & [nxd] -> [(>=m+n)xd])"""
    its = edgesIntersect(es1,es2)
    if len(its) == 0: 
        return np.concatenate([es1,es2],axis=0), np.concatenate([n1,n2],axis=0)
    in1 = arrContainBatch(np.concatenate(es1,axis=0),its)
    in2 = arrContainBatch(np.concatenate(es2,axis=0),its)
    if np.sum(in1) == len(its) and np.sum(in2) == len(its): # all are already cut
        return np.concatenate([es1,es2],axis=0), np.concatenate([n1,n2],axis=0)
    esout = []
    nout = []
    pt1 = its[in1==0]
    pt2 = its[in2==0]
    yin = batchBatchCoincident(es1,pt1,False)
    he = batchBatchCoincident(es2,pt2,False)
    yin = mergeEdgesHelper(esout, nout, es1, n1, pt1, yin)
    he = mergeEdgesHelper(esout, nout, es2, n2, pt2, he)
    return np.array(esout), np.array(nout)

def edgeToPointsSorted(sortedEdges): # fck
    """Helper function. Returns an array of points extracted from a sorted array of edges. Should be quite easy to figure out the algorithm."""
    return sortedEdges[:,0]

def edgeToPoints(edges): # ck
    """Returns an array of points extracted from any array of edges. edges in na.[nx2xd]"""
    rtn = []
    for i in range(len(edges)): 
        a0 = arrContainNorm(rtn,edges[i,0])
        if len(a0) == 0: 
            rtn.append(edges[i,0])
        a1 = arrContainNorm(rtn,edges[i,1])
        if len(a1) == 0:
            rtn.append(edges[i,1])
        # else both points are already in
    return np.array(rtn)

def isOutFlipped(connectingEdge, connectingEdgeOut, proposingEdge, proposingEdgeOut): 
    """Returns if proposingEdgeOut is potentially flipped. Computes based on angles of proposingEdge and proposingEdgeOut relative to connectingEdge, 
    positive toward connectingEdgeOut. If the proposingEdgeOut angle - 90deg does not meet the proposingEdge angle, the norm is probably flipped. 

    edges in [2xd], outs in [d]. Returns T/F and computed edge angle. 

    ***IF USING THE RESULT ONLY, TYPE isOutFlipped(...)[0]***
    """
    cnt = blurZero(np.linalg.norm(connectingEdge - ea1(proposingEdge),axis=-1))
    if len(cnt[cnt==0]) != 1: 
        print('isOutFlipped ERROR: No connection point found between two edges or two edges are identical.')
        print('connectingEdge:',connectingEdge)
        print('proposingEdge:',proposingEdge)
        raise AssertionError('Connecting edges must connect')
    cnt = np.argwhere(cnt==0)[0]
    notcnt = 1 - cnt
    ed = toUnit(connectingEdge[cnt[1]] - connectingEdge[notcnt[1]])
    en = connectingEdgeOut
    edi = toUnit(proposingEdge[notcnt[0]] - proposingEdge[cnt[0]])
    eni = proposingEdgeOut
    y = np.dot(edi,en)
    x = np.dot(edi,ed)
    egAng = np.arctan2(y,x)
    y = np.dot(eni,en)
    x = np.dot(eni,ed)
    normAng = np.arctan2(y,x)
    if np.dot(ang2Vec(egAng),ang2Vec(normAng-np.pi/2)) < 0:
        return True, egAng
    return False, egAng

def ptsInsideTheLane(es, dirs, pts, pas=1, nopas=-1e-12, edgePass=True, selfPass=False): # ck # Should be wrong now from a later point of view... But it works? Guess I'll just leave it alone...
    """Helper function. Identifies which points in pts are within the 'lane' of each edge in es + n*dirs (n>0). 
    Algorithm: Any inside must have the same sign: sign((e2-e1)·(pt-e1)) = sign((e1-e2)·(pt-e2)) (unless one of them is 0)
    
    es: [mx2xd], dirs: [mxd]_unit_vector, pts: [nxd]
    
    Returns a mask([nxm]) of if each point is inside, where mask[a,b] corresponds to pts[a] and es[b], and mask[a,b] = pas if inside, nopas if outside. 
    When pt[a] happens to be on an edge (INCLUDING COINCIDENT WITH AN EDGE POINT) of the lane es[b], mask[a,b] = pas if edgePass, nopas if not edgePass, edgePass if edgePass is overwriten to a number. 
    Same for selfPass, which controls weither the votices of edges (if included in pts) can be counted towards inside the lane.
    
    ***pas/nopas value setting are buggy. Please do after-processing instead in this version.***
    """
    eds = es[:,1] - es[:,0] # mxd
    ptss = np.expand_dims(pts,axis=1) # nx1xd
    d0 = ptss - es[:,0] # nx1xd - mxd -> nxmxd
    d1 = ptss - es[:,1]
    negiMsk = np.einsum('abc,bc->ab',d0,dirs) # nxm
    selfMsk = negiMsk * np.einsum('abc,bc->ab',d1,dirs)
    negiMsk = np.argwhere(negiMsk<-1e-11).T # [i_n, i_m].T
    
    dot0 = np.einsum('abc,bc->ab',d0,eds) # nxmxd * mxd -> nxm
    dot1 = np.einsum('abc,bc->ab',d1,-eds)
    perpMsk0 = np.argwhere(np.abs(dot0) < coincident_tol).T # [i_n, i_m].T
    perpMsk1 = np.argwhere(np.abs(dot1) < coincident_tol).T 
    selfMsk = -np.sign(np.abs(dot0*dot1)+selfMsk-1e-10) # '==0'
    selfMsk = np.argwhere(selfMsk==1).T
    blYne3 = np.sign(dot0) # nxm
    iiih23 = np.sign(dot1)
    inMsk = np.sign(np.abs(blYne3+iiih23))*pas # Only false condition: -1+1 = 0. 
    nop = np.argwhere(inMsk==0).T # [i_n, i_m].T
    inMsk[nop[0],nop[1]] = nopas

    if not isinstance(edgePass, bool): 
        inMsk[perpMsk0[0],perpMsk0[1]] = edgePass
        inMsk[perpMsk1[0],perpMsk1[1]] = edgePass
    elif edgePass: 
        inMsk[perpMsk0[0],perpMsk0[1]] = pas
        inMsk[perpMsk1[0],perpMsk1[1]] = pas
    else: 
        inMsk[perpMsk0[0],perpMsk0[1]] = nopas
        inMsk[perpMsk1[0],perpMsk1[1]] = nopas
    
    if not isinstance(selfPass, bool): 
        inMsk[selfMsk[0],selfMsk[1]] = selfPass
    elif selfPass: 
        inMsk[selfMsk[0],selfMsk[1]] = pas
    else: 
        inMsk[selfMsk[0],selfMsk[1]] = nopas
    
    inMsk[negiMsk[0],negiMsk[1]] = nopas
    return inMsk

def hasOutsideEdges(es, dirsOut, ckes=None, raw=False, removeSelf=True): # ck
    """Returns an array of indices [i_m] where corresponding edges have at least one outside edge(s). 
    Outside is defined by dirsOut (e.g. within boundary of es[i,0]+n*dirsOut[i] and es[i,1]+m*dirsOut[i], n,m > 0)
    
    es in [mx2xd] array of edges, dirsOut in [mxd] array of directions. Optional ckes as edges to check for in [ux2xd].
    If checked raw, outputs the covering mask used to obtain the result [mxm]/[mxu].

    Returns an array of indices [i_m] where corresponding edges have at least one outside edge(s). If ckes is None, default to check against es itself. If removeSelf set to True, will ignore the edge itself; if False, all those contained in ckes will count as having outside edges. 

    Algorithm is pretty much the same as ptsInsideTheLane. 

    ***NOTE IT MAY RECOGNIZE CONCAVE EDGES AS HAVING OUTSIDE EDGES*** 
    It is really hard to come up with an universal algorithm that won't run forever for that. So you do just have to either trust the sampling-based handler or do the case-specific after-processing yourself :/    """
    if ckes is None: 
        ckes = es
    # Intializing variables
    eds = es[:,1] - es[:,0] # mxd
    esa0 = ea1(es[:,0])
    esa1 = ea1(es[:,1])
    d00 = ckes[:,0] - esa0  # nxd - mx1xd -> mxnxd
    d10 = ckes[:,1] - esa0
    d01 = ckes[:,0] - esa1
    d11 = ckes[:,1] - esa1
    # Prepare self mask
    sm0 = 1.0 - np.sign(np.sign(np.linalg.norm(d00,axis=-1)-coincident_tol)+1) # Count self as outside
    sm1 = 1.0 - np.sign(np.sign(np.linalg.norm(d10,axis=-1)-coincident_tol)+1)
    sm2 = 1.0 - np.sign(np.sign(np.linalg.norm(d01,axis=-1)-coincident_tol)+1)
    sm3 = 1.0 - np.sign(np.sign(np.linalg.norm(d11,axis=-1)-coincident_tol)+1)
    selfMsk = np.sign((1-sm0)*(1-sm1) + (1-sm2)*(1-sm3)) # An edge that is itself has 0 in both sm regions
    # Prepare negative mask
    negiMsk0 = np.sign(blurZero(np.einsum('abc,ac->ab',d00,dirsOut))) # mxn
    negiMsk0[negiMsk0<coincident_tol] = 0
    negiMsk1 = np.sign(blurZero(np.einsum('abc,ac->ab',d10,dirsOut))) # Has to have at least 1 point > 0, thus sum >= 0 (0,1; -1,1; 0,0; 1,1)
    negiMsk1[negiMsk1<coincident_tol] = 0
    # Core: Two points must be on the different side of at least one point, thus, dot00 + dot10 = 0 or dot01 + dot11 = 0
    dot00 = np.sign(blurZero(np.einsum('abc,ac->ab',d00,eds))) - sm0 # mxnxd * mxd -> mxn
    dot10 = np.sign(blurZero(np.einsum('abc,ac->ab',d10,eds))) - sm1 # exception of perp,
    dot01 = np.sign(blurZero(np.einsum('abc,ac->ab',d01,-eds))) - sm2 # which are counted wherever the other one is,
    dot11 = np.sign(blurZero(np.einsum('abc,ac->ab',d11,-eds))) - sm3 # as long as the perp point is not self
    c0 = np.abs((dot00 + dot10)//2) # => ..+.. = 1 or -1 are also counted
    c1 = np.abs((dot01 + dot11)//2) # Just can't be on the same side
    inMsk = 1.0 - c0 * c1 # mxn
    # Post processing to apply negative mask and self mask
    negiMsk = negiMsk0*negiMsk1
    coin0 = batchBatchCoincident(ckes,es[:,0],False).T
    coin1 = batchBatchCoincident(ckes,es[:,1],False).T
    negiMsk += np.sign(negiMsk0+negiMsk1)*(1-np.sign(coin0+coin1))  # Must be: (negiMsk0 > 0 and negiMsk1 > 0) 
    c10 = coin0*np.max([dot10,np.zeros_like(dot10)],axis=0)         # or (negiMsk0 <= 0 and negiMsk1 > 0 and ((pass through 0 and dot10 > 0) or (pass through 1 and dot11 > 0)))
    c11 = coin1*np.max([dot11,np.zeros_like(dot11)],axis=0)         # or (negiMsk1 <= 0 and negiMsk0 > 0 and ((pass through 0 and dot00 > 0) or (pass through 1 and dot01 > 0))) 
    c00 = coin0*np.max([dot00,np.zeros_like(dot00)],axis=0)         # or (negiMsk0 > 0 or negiMsk1 > 0) and (not passing through any of the points) 
    c01 = coin1*np.max([dot01,np.zeros_like(dot01)],axis=0)         # Theory: If any one of the points is not on the outside, the other point must be outside and inside direction
    negiMsk += (1-negiMsk0)*negiMsk1*np.sign(c10+c11)
    negiMsk += (1-negiMsk1)*negiMsk0*np.sign(c00+c01)
    negiMsk = np.sign(negiMsk)
    inMsk *= negiMsk
    if removeSelf:
        inMsk *= selfMsk
    else:
        inMsk = np.sign(inMsk + 1 - selfMsk)
    inMsk = np.sign(np.sum(inMsk,axis=1))
    # Handling part of concave shapes
    samples = eqCircularSample(8*vis_sample_pts)
    tes = es[inMsk==1] # The algorithm garantees no false negative, thus handling false positive only.
    tn = dirsOut[inMsk==1]
    mids = (tes[:,1]+tes[:,0])/2 + tn*3*coincident_tol
    yinheOnBili = np.ones(len(tes))
    temp = np.zeros_like(samples)
    for i in range(len(tes)): 
        temp[:] = mids[i]
        pt, test = batchIntersect(temp,samples,ckes)
        test[test==-1] = 0
        test = np.sign(test)
        yinheOnBili[i] = np.min(np.sum(test,axis=-1)) # As long as any 1 direction is not covered, this edge is not covered by an outside edge.
    inMsk[inMsk==1] *= yinheOnBili
    if raw: 
        return inMsk
    return np.argwhere(inMsk>0).T[0]

def edgeDirs(edges): # fck
    """Helper function. Return directions of edges [nx2xd]"""
    yinhe233 = edges[:,1] - edges[:,0]
    l = np.linalg.norm(yinhe233,axis=-1)
    return yinhe233 / dimUp(l)

def overlapWhich(lst, edge, nop = np.array([-1])): # ck
    """Helper function. Returns which index(s) in lst does edge has an overlapping point. Returns nop if doesn't.
    lst in mx[_x2xd], edge in [2xd]"""
    if len(lst) == 0: 
        return nop
    # Well it can only be O(n)
    rtn = []
    for i in range(len(lst)): 
        l = lst[i]
        if len(l) == 0: 
            continue
        ph3536 = np.concatenate(l,axis=0)
        dis0 = np.linalg.norm(ph3536 - edge[0], axis=-1)
        dis1 = np.linalg.norm(ph3536 - edge[1], axis=-1)
        if np.sum(np.sign(np.abs(dis0*dis1))) < len(ph3536): 
            rtn.append(i)
    if len(rtn) == 0: 
        return nop
    return np.array(rtn)

def ptsToEdges(pts): # fck
    """Helper function. Convert array of points [nxd] into array of edges [nx2xd]. Assumes points are aranged in some order that makes the result valid."""
    es = np.zeros([len(pts),2,pts.shape[-1]])
    es[:,0] = pts
    es[:-1,1] = pts[1:]
    es[-1,1] = pts[0]
    return es

def batchPtsToEdges(ptss): # ck
    """ptsToEdges but [nxmxd] -> [nxmx2xd], wraps around on the second axis (n)"""
    s = ptss.shape
    es = np.zeros([s[0],s[1],2,s[-1]])
    es[:,:,0] = ptss
    es[:,:-1,1] = ptss[:,1:]
    es[:,-1,1] = ptss[:,0]
    return es

def batchCoincident(edge, pts, mathematical=True): # ck
    """Compute if all pt in pts are coincident with edge. Algorithm native to 2D
    
    edge in [2xd] array of vortices, pts in [nxd] array of points. 
    If mathematical is set to True, those along the direction of edge but are outside of edge's range are also counted (so like coincident in any CAD)
    If set to False, will be using geometrical meaning of coincident.
    
    Returns a mask [n] of which points are coincident (1=True, 0=False)"""
    edir = edge[1] - edge[0]
    en = inPlaneNorm(edir)
    pr = pts-edge[0]
    rua = np.abs(np.dot(pr,en))
    msk = np.ones(len(pts))
    msk[rua>coincident_tol] = 0 
    if mathematical:
        return msk
    el = np.linalg.norm(edir)
    edir /= el
    oiai = blurZero(np.dot(pr, edir))
    msk[oiai<-coincident_tol] = 0
    msk[oiai>el+coincident_tol] = 0
    return msk

def batchBatchCoincident(edges, pts, mathematical=True): # ck
    """Same as batchCoincident except supports multiple edges. edges in [mx2xd] and pts in [nxd]. Returns a mask [mxn]. 2D only."""
    edir = edges[:,1] - edges[:,0]
    en = np.linalg.norm(edir,axis=-1)
    edir /= dimUp(en) # mxd
    edn = inPlaneNorm(edir) # mxd
    pr = pts-ea1(edges[:,0]) 
    rua = np.abs(np.einsum('mnd,md->mn',pr,edn)) # mxn
    msk = np.ones([len(edges),len(pts)])
    msk[rua>coincident_tol] = 0
    if mathematical:
        return msk
    oiai = blurZero(np.einsum('mnd,md->mn',pr,edir))
    msk[oiai<-coincident_tol] = 0
    msk[oiai>dimUp(en+coincident_tol)] = 0
    return msk

def batchBatchCoincident3D(edges, pts, mathematical=True): # ck
    """Same as batchCoincident except supports multiple edges. edges in [mx2xd] and pts in [nxd]. Returns a mask [mxn]. Can do 3D."""
    edir = edges[:,1] - edges[:,0]
    en = np.linalg.norm(edir,axis=-1)
    edir /= dimUp(en) # mxd
    edn = getANorm(edir) # mxd
    pr = pts-ea1(edges[:,0]) 
    rua = np.abs(np.einsum('mnd,md->mn',pr,edn)) # mxn
    msk = np.ones([len(edges),len(pts)])
    msk[rua>coincident_tol] = 0
    if mathematical:
        return msk
    oiai = blurZero(np.einsum('mnd,md->mn',pr,edir))
    msk[oiai<-coincident_tol] = 0
    msk[oiai>dimUp(en+coincident_tol)] = 0
    return msk

def edgeEqual(e1, e2): # fck
    """Returns if e1 is equal to e2. True for flipped edges."""
    d1 = e1 - e2
    if -coincident_tol < np.sum(np.linalg.norm(d1,axis=-1)) < coincident_tol: 
        return True
    d2 = e1 - np.flip(e2,axis=0)
    if -coincident_tol < np.sum(np.linalg.norm(d2,axis=-1)) < coincident_tol: 
        return True
    return False

def edgeContain(edges, edge, raw=False): # ck
    """Returns if edge is already contained in edges. Returns true for flipped vortices. If checked raw, outputs the raw mask used to compute result.
    
    edges in [nx2xd] array of edges, edge in [2xd] array of vortices. Optional return raw in [n]. Note you may receive False when len(edges) == 0"""
    if len(edges) <= 0: 
        return False
    d1 = edges - edge
    d2 = edges - np.flip(edge,axis=0)
    r1 = np.sum(blurZero(np.linalg.norm(d1,axis=-1)),axis=-1)
    r2 = np.sum(blurZero(np.linalg.norm(d2,axis=-1)),axis=-1)
    cyrene = r1 * r2
    cyrene = np.sign(cyrene)
    if raw: 
        return cyrene # Please :((
    re1l = np.sum(cyrene) # n -> 1
    return re1l < len(edges)

def edgeContainBatch(edges1, edges2): # ck
    """Batch version of edgeContain. Returns a mask corresponding to each element in edges2, where mask[i] == 1 if edges2[i] is contained in edges1, 0 otherwise.
    
    edges1 in [nx2xd], edges2 in [mx2xd], returns a mask in [m]"""
    if len(edges1) <= 0: 
        return np.zeros(len(edges2))
    if len(edges2) <= 0: 
        return np.array([])
    d1 = edges1 - ea1(edges2)
    d2 = edges1 - ea1(np.flip(edges2,axis=1))
    r1 = np.sum(blurZero(np.linalg.norm(d1,axis=-1)),axis=-1)
    r2 = np.sum(blurZero(np.linalg.norm(d2,axis=-1)),axis=-1)
    freeCode = np.sign(r1*r2)
    return 1 - np.min(freeCode,axis=-1)

def edgeContain2(edges, edge): # fck
    """Returns if edge is fully covered by an edge in edges. 
    E.g. edges = [...,[[0,0],[0,2]],...], edge = [[0,0.1],[0,1.2]] -> True; 
    Will NOT return True if edge is only covered by multiple edge in edges. E.g. edges = [[[0,0],[0,1]], [[0,1],[0,2]]], edge = [[0,0.5],[0,1.5]] -> False"""
    rst = batchBatchCoincident(edges, edge, mathematical=False)
    return np.max(np.sum(rst,axis=-1)) >= 2

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Triangle (mesh) operations

def trigs2Edges(trigs): # ck
    """nx3xd [[pt1,pt2,pt3],...] -> 3nx2xd [[pt1,pt2],[pt2,pt3],[pt3,pt1],...]"""
    s = trigs.shape
    rtn = np.zeros([s[0],3,2,s[-1]])
    rtn[:,0,0] = trigs[:,0]
    rtn[:,0,1] = trigs[:,1]
    rtn[:,1,0] = trigs[:,1]
    rtn[:,1,1] = trigs[:,2]
    rtn[:,2,0] = trigs[:,2]
    rtn[:,2,1] = trigs[:,0]
    return rtn.reshape((3*s[0],2,-1))

def inTrig(trig, pt, edgePass=True): # ck
    """Returns if the triangle on this plane contains this point. Trig in [[a*e1,b*e2],...], pt in [a*e1,b*e2]. Algorithm from https://zhuanlan.zhihu.com/p/106253152"""
    ds = pt - trig
    r1 = blurZero(np.cross(ds[0], ds[1]),coincident_tol)
    r2 = blurZero(np.cross(ds[1], ds[2]),coincident_tol)
    r3 = blurZero(np.cross(ds[2], ds[0]),coincident_tol)
    if r1 == 0: 
        if not edgePass: 
            return False
        el = np.linalg.norm(trig[1]-trig[0])
        return np.linalg.norm(ds[0]) < el and np.linalg.norm(ds[1]) < el
    if r2 == 0: 
        if not edgePass: 
            return False
        el = np.linalg.norm(trig[2]-trig[1])
        return np.linalg.norm(ds[2]) < el and np.linalg.norm(ds[1]) < el
    if r3 == 0: 
        if not edgePass: 
            return False
        el = np.linalg.norm(trig[2]-trig[0])
        return np.linalg.norm(ds[0]) < el and np.linalg.norm(ds[2]) < el
    return np.sign(r1) == np.sign(r2) == np.sign(r3)
    
def batchInTrig(trig, pts): # ck
    """Batch check if a bunch of points are inside a trig. trig in [3xvortices(2D)], pts in [nx2]. Returns a masking array [n], where arr[i]=1 if pt[i] is inside and arr[i]=0 if not"""
    ds = np.expand_dims(trig,axis=1) - pts
    rst = np.array([np.cross(ds[0],ds[1]),np.cross(ds[1],ds[2]),np.cross(ds[2],ds[0])])
    rst = np.sign(rst)
    mski = np.sum(np.sign(rst+1e-6),axis=0) # Make sure 0 is counted towards good
    mskj = np.sum(np.sign(rst-1e-6),axis=0)
    msk = np.zeros_like(mski)
    msk[mski==3] = 1
    msk[mski==-3] = 1
    msk[mskj==3] = 1
    msk[mskj==-3] = 1
    return msk

def onTrigs(trigs, pt): # ck
    """Returns if pt is on one of the vortices or edges of any trig in trigs. trigs in [nx3xd], pt in [d]. Natively developed for 3D.
    Also returns relaTrigs and selfBatchCross(trigs_relative_to_pt) to avoid repeating calculations. (If the result is True, both will be an empty array)"""
    # On vortex check
    relaTrigs = trigs - pt
    vortt = blurZero(np.linalg.norm(relaTrigs,axis=-1))
    if len(vortt[vortt==0]) > 0: 
        return True, np.array([]), np.array([])
    
    # On edge condition: any two relative dirs cross to be 0, and dot be <0
    crs = selfBatchCross(relaTrigs)
    crossMsk = 1 - np.sign(blurZero(np.linalg.norm(crs,axis=-1)))
    dotMsk = -np.sign(blurZero(selfBatchDot(relaTrigs)))
    dotMsk[dotMsk<0] = 0
    if np.sum(crossMsk*dotMsk) > 0: 
        return True, np.array([]), np.array([])
    
    # On surface condition: three cross vectors colinear, self dot no -1
    crsDirs = toUnit(crs)
    colin = selfBatchCross(crsDirs)
    colinMsk = 1 - np.sign(blurZero(np.sum(np.abs(colin),axis=(-1,-2))))
    dotMsk = np.sum(np.sign(blurZero(selfBatchDot(crsDirs))),axis=-1)
    dotMsk[dotMsk<3] = 0
    dotMsk = np.sign(dotMsk)
    return np.sum(colinMsk*dotMsk) > 0, relaTrigs, crsDirs

def shadedInTrigs(trigs, start, dirs, raw=False): # ck
    """Batch check if all dirs from start are covered by at least 1 trig in trigs. 
    Trigs in np.array([[[x,y,z]_1,[x,y,z]_2,[x,y,z]_3],...]) (nx3x3), start in np.array([x,y,z]) (3), dirs in np.array([[x,y,z]_1,...]) (vx3)
    
    Basically project all trigs on the positive direction to the start's plane (perp to dir, for each dir at the same time) and check if start is in any trig in trigs

    If raw is set to True, returns the rst mask intstead.
    """
    if trigs is None or len(trigs) < 1: 
        print('shadedInTrigs ERROR: Trig mesh is None. Please double check your file. If you confirmed that your file is okay, please report to the author.')
        raise AssertionError('TrigMesh must have a trig')
    if len(start) < 3: 
        print('You forgot to convert test point to 3D.')
        raise AssertionError('^^')
    # Simple skip test 'cause full computing takes a long time and can't handle certain edge cases
    skipTest, relTrigs, crs = onTrigs(trigs,start)
    if skipTest: 
        if raw: 
            return np.array([0])
        return True

    # Batch compute conditions & take sign only (~T/F). Conditions: inTrig (cross prods on the same direction), and on the positive direction
    nd, vali = batchIntersect3d(start, dirs, trigs) # Intersection point on positive direction, # trigs x dirs
    if not fast_intersect: # Those results are supposed to be invalid... But I guess the way I applied validation mask during calculation made the result not that invalid...
        nd *= vali # Anyway, another story of 'how did it work'...
    # nd = np.min([nd,np.ones_like(nd)*3],axis=0)
    # nd = np.max([nd,-np.ones_like(nd)*3],axis=0)
    hpt = start + dimUp(nd) * dirs # trigs x dirs x 3
    relTrigs = ea1(trigs) - np.expand_dims(hpt,axis=2) # trigs x 1 x 3 x 3 - t x d x 1 x 3
    relTrigs = np.reshape(relTrigs, (-1,3,3)) # -> (t*d)x3x3
    bcc = toUnit(blurZero(selfBatchCross(relTrigs)))
    dcc = np.sign(blurZero(selfBatchDot(bcc))) # (t*d)x3x3 -> (t*d)x3
    dcc = np.reshape(dcc, (len(trigs),-1,3)) 
    intrigMsk = np.sign(np.min(dcc,axis=-1) + 1) # Dot prod must all be 1 to be inside trig, or has a 0 to be on edge
    posHitMsk = np.sign(nd) # On positive direction mask
    posHitMsk[nd < 0] = 0
    if fast_intersect:
        confusedMsk = dimUp(crs[:,0,0] * (start - ea1(start))[0,0]) * vali
        rst = np.sum(posHitMsk * (intrigMsk + confusedMsk), axis=0)
    else: 
        rst = np.sum(posHitMsk * (intrigMsk), axis=0)
    # es, cs = dirHandlerAnalog(start, dirs, rst)
    # plot3D(trigs, es, surfaceColors=np.array(['black']), edgeColors=cs, scatterPoint=hpt, title='gugugaga')
    if raw:
        return rst
    return len(rst[rst<=0]) <= 0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Rectangle utilities

def rectContainStrict(rects, rect): # ck
    """Helper function. Returns a mask that mask[i] = 1 where rects[i] == rect for all vortices, 0 else. rects [nx4xd], rect [4xd] -> mask [n]. Does handle points in different order."""
    diff = np.sum(np.min(blurZero(np.sum(np.abs(np.expand_dims(rects,axis=2) - rect),axis=-1),coincident_tol),axis=1), axis=(-1))
    return 1 - np.sign(diff)

def rectContain(rects, rect): # ck
    """Returns a mask of where rect is contained in rects. (in geometrical meaning -- 1 if rect is fully covered by a rectangle in rects) (1 for contained, 0 for false)
    
    rects in mx[4xd] list of rectangles, rect in [4xd] array of vortice points. 
    
    Algorithm: Compute norm of rect -> Find if any rect in rects have intersection from each point to direction of both norms. rects in different orders are recoginized.
    Will count self if contained."""
    recs = np.array(rects)
    ed1 = rect[1] - rect[0]
    ed2 = rect[3] - rect[0]
    n1 = np.linalg.norm(ed1)
    n2 = np.linalg.norm(ed2)
    ed1 /= n1
    ed2 /= n2
    # To relative coords
    recs -= blurZero(rect[0],coincident_tol)
    recsx = blurZero(np.dot(recs,ed1)) / n1
    recsy = blurZero(np.dot(recs,ed2)) / n2
    relaRecs = blurZero(np.stack([recsx,recsy],axis=-1))
    relaRecEgs = batchPtsToEdges(relaRecs) 
    # Compute norm & intersections
    nsd = np.array([[0,-1],[-1,0],[-1,0],[0,1],[0,1],[1,0],[1,0],[0,-1]])
    c = 1e-9
    ptsc = np.array([[0+c,0+c],[0+c,0+c],[0+c,1-c],[0+c,1-c],[1-c,1-c],[1-c,1-c],[1-c,0+c],[1-c,0+c]]) # Declare slightly smaller to avoid 0 issues
    msk = np.zeros(len(rects))
    for i in range(len(relaRecEgs)): 
        ret = relaRecEgs[i]
        _,n = batchIntersect(ptsc,nsd,ret)
        n[n==0] = 1
        n[n==-1] = 0
        y_inh_e_23_3 = np.sign(np.sum(n,axis=-1))
        a_u_th = np.sum(y_inh_e_23_3)
        if a_u_th == 8: 
            msk[i] = 1
    return msk

def rectLengths(rect): # fck
    """Returns two edge lengths of the rectangle. rect in [4xd]. Returns [2] np.array not sorted"""
    d1 = np.linalg.norm(rect[1] - rect[0])
    d2 = np.linalg.norm(rect[3] - rect[0])
    if np.abs(d1-d2) < coincident_tol: 
        d2 = np.linalg.norm(rect[2] - rect[0])
    return np.array([d1,d2])

def smallestCoveringRect(poly, returnAll=False): # ck
    """Generate the smallest (area-wise) rectangle that covers the entire shape. Rectangle can rotate to any angle. 
    input poly in [nx2xd] array of edges, output [4xd] vortices of smallest covering rectangle and the convex hull in its edges [mx2xd]
    Uses a convex hull -> co-edge rectangle algorithm inspired by https://zhuanlan.zhihu.com/p/679819021
    Is obviously a 2D-only algorithm
    Optional returnAll makes the function return a sorted array of all possible smallest covering rectangles, from smallest area to biggest, if set True. Returns the smallest one by default."""
    vorts = edgeToPoints(poly)
    cvhv = toConvexHull(vorts)
    cvh = ptsToEdges(cvhv)
    cvhd = cvh[:,1]-cvh[:,0]
    cvhd /= dimUp(np.linalg.norm(cvhd,axis=-1))
    p2p = cvhv - ea1(cvhv)
    # Find max height
    norm = inPlaneNorm(cvhd)
    heights = np.einsum('nmd,nd->nm',p2p,norm) 
    abd = np.abs(heights)
    maxdi = np.argmax(abd, axis=1)
    maxd = abd[np.arange(len(heights)),maxdi]
    # Find max length
    lengths = np.einsum('nmd,nd->nm',p2p,cvhd)
    m1 = np.argmax(lengths, axis=1)
    m2 = np.argmax(-lengths, axis=1)
    ml1 = lengths[np.arange(len(lengths)),m1]
    ml2 = lengths[np.arange(len(lengths)),m2]
    mlEs = transpose2D(np.array([cvhv+dimUp(ml1)*cvhd,cvhv+dimUp(ml2)*cvhd])) # 2xnxd -> nx2xd
    ml = np.linalg.norm(mlEs[:,1]-mlEs[:,0],axis=-1)
    area = ml * maxd
    minRi = np.argsort(area)
    if returnAll: 
        otherSide = mlEs[minRi] + ea1(dimUp(heights[minRi,maxdi[minRi]])*norm[minRi])
        return blurZero(transpose2D(np.array([mlEs[minRi,0], otherSide[:,0], otherSide[:,1], mlEs[minRi,1]])))
    minRi = minRi[0]
    otherSide = mlEs[minRi] + heights[minRi,maxdi[minRi]]*norm[minRi]
    return blurZero(np.array([mlEs[minRi,0], otherSide[0], otherSide[1], mlEs[minRi,1]]))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Polygon processing functions

def sortPoly(poly, norm=None): # ck
    """Sorts a polygon (an array of edges, [nx2xdim]) into an order. The sorted polygon will garantee poly[i,1] == poly[i+1,0] for any i in range(len(poly)-1)

    ASSUMES no hole and the poly is manifold. Returns sorted poly in [nx2xdim] and sorted norm in [nxdim] (empty list if no norm is put in)
    
    You can attach an norm [nxdim] to be sorted with poly. Assumes norm is index-wise already alighed with poly and is in the direction you want. 
    If an norm is attached, the return will be a list where [0] is the sorted poly and [1] is the sorted norm."""
    bi38li83c3h3e5e7r = np.reshape(poly,(-1,poly.shape[-1])) # 2nxd
    diff = np.abs(blurZero(np.expand_dims(bi38li83c3h3e5e7r,axis=1) - bi38li83c3h3e5e7r)) # 2nx2nxd
    diff = np.sum(diff,axis=-1) # 2nx2n -> vxm
    eqs = []
    for i in range(len(diff)): 
        eqs.append(np.argwhere(diff[i]==0)) # nx[2x[i_m]]
    sorted = []
    sortedNorm = []
    through = 0
    idx = 0
    while through < len(poly): 
        through += 1
        if idx % 2 == 0:
            sorted.append(poly[idx//2])
            if norm is not None:
                sortedNorm.append(norm[idx//2])
        else: 
            sorted.append(np.flip(poly[idx//2],axis=0))
            if norm is not None:
                sortedNorm.append(norm[idx//2])
        idx = otherone(idx)
        for j in range(len(eqs[idx])): 
            if eqs[idx][j,0] != idx: 
                idx = eqs[idx][j,0]
                break
            if j == len(eqs[idx])-1 and through < len(poly): 
                print('sortPoly: wtf? No continuing @idx =',idx)
                return []
    return np.array(sorted), np.array(sortedNorm)

def isConvex(polyPoints, outside): # 'f'ck
    """Helper function. Returns if the poly is convex. polyPoints in [nxdim] array of vortices, norm in [nxdim] corresponding to poly index-wise"""
    rela = np.expand_dims(polyPoints,axis=1) - polyPoints # nxnxdim
    oOoOoOoOO = np.sign(np.sign(np.einsum('bcd,cd->bc',rela,outside))-0.01) # Exception to those on the edge, which are allowed
    yorihe = -np.sum(oOoOoOoOO,axis=-1) # Any point inside must not have any positive dot with any outside vector
    return all(i==len(polyPoints) for i in yorihe)

def isConvexSorted(sortedPoly): # ck
    """Helper function. Returns if sortedPoly is convex. Sorted poly in [nx2xd] array of edges, sorted in one direction. (2D)"""
    ed = sortedPoly[:,1] - sortedPoly[:,0]
    edr = shiftRight(ed.T).T
    v5gE8YqdNm1b = np.sum(np.sign(np.cross(ed,edr)))
    return v5gE8YqdNm1b == len(sortedPoly) or v5gE8YqdNm1b == -len(sortedPoly)

def removeDiagonal(mtx): # ck
    """Helper function. Returns a new matrix that removes diagonal elements of mtx. If mtx is more than 2D, used the first two dimensions only. Shrinks along the second axis (axis index 1)
    Assumes mtx is square. If mtx is not square, the function will run but the result may be weird."""
    if len(mtx) == 0 or mtx.ndim<2 or len(mtx[0]) == 0: 
        return np.array([])
    spe = np.array(mtx.shape)
    spe[1] -= 1
    rtn = np.zeros(spe)
    l = np.min([len(mtx),len(mtx[0])])
    for i in range(l): 
        if i > 0: 
            rtn[i,:i] = mtx[i,:i]
        if i < len(mtx)-1: 
            rtn[i,i:] = mtx[i,i+1:]
    return rtn

def edgeIntersect(eg1,eg2): # ck
    """Returns intersection points of two edges. eg1 and eg2 in [2xd]. Assumes they do have an intersection within reasonable range."""
    d1 = eg1[1] - eg1[0]
    d1n = -1*d1
    ds = toUnit(np.array([d1,d1n]))
    pts = np.array([eg1[0],eg1[0]])
    d2 = toUnit(eg2[1] - eg2[0])
    eg2inf = np.array([[eg2[0]-d2*1145141919,eg2[1]+d2*1145141919]])
    its, n = batchIntersect(pts,ds,eg2inf)
    ni = np.argmax(n)
    return its[ni,0]

def edgesIntersect(egs1,egs2): # ck
    """Batch version of edgeIntersect. egs1 in [nx2xd], egs2 in [mx2xd]. Returns an array of all intersection points between each edges of the two [kxd]
    If none or any egs is empty, returns an empty array. Will NOT assume infinite length this time."""
    if len(egs1)==0 or len(egs2)==0: 
        return np.array([])
    assert egs1.shape[-1] == egs2.shape[-1]
    pts = egs1[:,0]
    dirs = egs1[:,1] - egs1[:,0]
    its, n = batchIntersect(pts,dirs,egs2)
    n[n>1+coincident_tol] = -1
    return removeIdentical(its[n!=-1])

def expandPoly(sortedPoly, sortedOutsideNorm, expansion): # ck
    """Helper function. Expands sortedPoly to the direction of sortedOutsideNorm by amount of expansion. Doing so by vector summation."""
    raw = sortedPoly + ea1(expansion * sortedOutsideNorm)
    yh = np.zeros_like(raw)
    for i in range(0,len(raw)-1): 
        yh[i,0] = edgeIntersect(raw[i-1],raw[i])
        yh[i,1] = edgeIntersect(raw[i],raw[i+1])
    yh[-1,0] = edgeIntersect(raw[-2],raw[-1])
    yh[-1,1] = edgeIntersect(raw[-1],raw[0])
    return yh

def traceColinearEdges(edges, edgei): # ck
    """Returns a mask which edge in edges are colinear to and connected with edgei (and do not form an intersection after merging), where edgei is the index of starting edge in edges."""
    msk = np.zeros(len(edges))
    pts = np.reshape(edges,(-1,edges.shape[-1]))
    cntMsk = 1 - np.sign(blurZero(np.linalg.norm(pts - ea1(pts),axis=-1)))
    cntMsk[np.sum(cntMsk,axis=-1) > 2] = 0
    colinMsk = np.sign(np.sum(batchBatchCoincident(edges,edges[edgei],mathematical=True),axis=1)//2)
    cntMsk *= dupArr(colinMsk)
    cntMsk = 1 - cntMsk
    traceEdgeCore(edgei*2,msk,cntMsk)
    return msk

def removeExcessEdges(edgeList, edgeNormList): # ck
    """Helper function. Merges (!!!REMOVES ORIGINAL EDGES!!!) all colinear edges in edgeList (and corresponding norms in edgeNormList), given they do not intersect with other edges after merging.
    Works by mutating list. No return."""
    es = np.copy(edgeList)
    coinMsk = np.zeros(len(es))
    colinegs = []
    colinnorms = []
    ctr = 0
    for i in range(len(es)): 
        if coinMsk[i] > 0: 
            continue
        y2h3i3en = traceColinearEdges(es,i)
        if np.sum(y2h3i3en) > 1:
            colinegs.append([])
            colinnorms.append(edgeNormList[i])
            ee = es[y2h3i3en==1]
            for ej in ee:
                colinegs[ctr].append(ej)
            coinMsk += y2h3i3en
            ctr += 1

    for i in range(len(es)-1,-1,-1): 
        if coinMsk[i] >= 1:
            del edgeList[i]
            del edgeNormList[i]

    for i in range(len(colinegs)): 
        ei = concateEdges(np.array(colinegs[i]))
        edgeList.append(ei)
        ni = inPlaneNorm(ei[1]-ei[0])
        if np.dot(ni,colinnorms[i]) < 0: 
            ni *= -1
        edgeNormList.append(ni)

def uniformSample(poly, outside): # ck
    """Returns a uniform (rectangular) sample point (2D) in poly according to min_edge_length and max_samples in config (1 sample point per min_edge_length, maximum max_samples). 
    If poly is too small to have a sample point, returns an empty array.

    poly in np.array([mx2x2]) list of edges. Returns a list of sample points ([nx2]). In case of 1 sample only, n=1. 
    When isRect is set to True, uses an alternative (a lot faster) pipeline that samples across the rectangle formed by poly[0] and poly[1] only.

    Computes by generating a mesh over the smallest and biggest x and y values of poly and return only those inside.
    Just convert to trigs and iterate over inside any trigs"""
    minx = np.min(poly[...,0])
    miny = np.min(poly[...,1])
    maxx = np.max(poly[...,0])
    maxy = np.max(poly[...,1])
    nhe = int((maxx-minx)//min_edge_length)
    ier = int((maxy-miny)//min_edge_length)
    if nhe == 0 or ier == 0: 
        return np.array([])
    if vis_sample_pts <= 1: 
        ctn = np.array([(minx+maxx)/2, (miny+maxy)/2])
        if inpoly(poly,outside,ctn): 
            return np.array([ctn])
        else: 
            b3835, _ = sortPoly(poly)
            d8337 = np.linalg.norm(b3835[:,0] - ctn, axis=-1)
            return b3835[np.argmin(d8337,axis=0),0]
    xsamples = np.linspace(minx,maxx,np.min([nhe,vis_sample_pts]))
    ysamples = np.linspace(miny,maxy,np.min([ier,vis_sample_pts]))
    x,y = np.meshgrid(xsamples,ysamples)
    # HTF am I being chased by the inside problem all the time? 
    meshRaw = np.stack((x,y),axis=-1)
    meshRaw = np.reshape(meshRaw,(-1,2))
    yi2nh3e3 = toTrigsOneGo(poly, outside)
    msk = np.ones(meshRaw.shape[0])
    for trig in yi2nh3e3: 
        msk *= 1-batchInTrig(trig,meshRaw)
    return meshRaw[msk==0]

def uniformSampleRect(rect): # ck
    """Uniform sampling function for rectangles only. rect in nxmxd array of vortices (in along-x_along-y_coord)"""
    i33 = rect[1] - rect[0]
    d57 = rect[3] - rect[0]
    b3883 = rect[0]
    if np.abs(np.dot(i33,d57)) > coincident_tol: # In case of not-that-legal rectangle
        d57 = rect[2] - rect[0]
    n1 = np.linalg.norm(i33)
    n2 = np.linalg.norm(d57)
    i33 /= dimUp(n1)
    d57 /= dimUp(n2)
    xss = int(n1//min_edge_length)
    yss = int(n2//min_edge_length)
    if xss == 0 or yss == 0: 
        return np.array([])
    ptsx = np.min([xss,vis_sample_pts])
    ptsy = np.min([yss,vis_sample_pts])
    sampleBasex = np.arange(ptsx)
    xdis = n1 / ptsx
    sampleBasey = np.arange(ptsy)
    ydis = n2 / ptsy
    xsamples = sampleBasex * xdis + xdis/2
    ysamples = sampleBasey * ydis + ydis/2
    x,y = np.meshgrid(xsamples,ysamples)
    autYinhe = b3883 + dimUp(x)*i33 + dimUp(y)*d57
    return autYinhe

def toConvexCore(sortedPoly, sortedOutsideNorm, recr): # ck
    """Core function. Converts a polygon (a list of 2D edges, [nx2x2]) into a list of convex polygons [vx[_x2x2]]. recr is an internal variable.  
    
    If the polygon is already convex, returns itself [nx2x2].
    
    Algorithm: 
        For all point in poly, try to launch lines to all other points
        Compute batch intersect of *slightly enlarged* edges
        Cut the poly into 2 at a position that will not cross the poly boundary
        Recursion on both pieces until all pieces are convex"""
    if len(sortedPoly) <= 2: 
        print('\033[91mtoConvexCore ERROR: Illegal argument: The input sortedPoly has length <= 2.\033[0m This should not happen with any reasonable obj file. You may have a broken data.')
        raise AssertionError('Poly must have >=3 edges')
    
    if len(sortedPoly) <= 3 or isConvexSorted(sortedPoly):
        recr.append(sortedPoly)
        return
    pts = edgeToPointsSorted(sortedPoly) # nxd
    lpt = len(pts)
    norms = sortedOutsideNorm
    # Form lines -- not really lines just directions
    dirs = pts-ea1(pts) # nxn -> nx(n-1)
    temp = np.zeros([lpt,2]) # (n-1)x2
    
    # Batch intersect
    esl = expandPoly(sortedPoly,sortedOutsideNorm,10*coincident_tol)
    ns = np.zeros([lpt,len(esl),len(dirs[0])]) # nxexdr
    for i in range(lpt):
        temp[:] = pts[i]
        its, n = batchIntersect(temp,dirs[i],esl) # (n/dr)xe. For a convex shape, this should be all -1 or > between point distance (>1)
        n[n>=1+1e-10] = -1
        ns[i] = n.T

    # Find a good point to cut and form new polys. Recursion on new polys until all are convex. 
    hits_e = np.sum(np.sign(ns),axis=1) # nxdr
    hits_pt = np.sum(hits_e,axis=-1) # n
    bestPt = np.argmin(hits_pt, axis=0) # Garanteed can-do: Any polygon is approaximateble by trigs, thus must have at least one point that can see at least two other points clearly. Therefore, garanteed can split at the best point. 
    bufferi = False
    bufferj = False
    normsOuti = []
    normsOutj = []
    coini = []
    oct = 0
    i = circularAddInt(bestPt, lpt//2, lpt)
    while True: 
        if oct > lpt: 
            print('\033[91mERROR: toConvexCore failed to find cutting point of a polygon (Type I).\033[0m This should never happen. Please report to the author.')
            # plotEdges(sortedPoly,0.5*sortedOutsideNorm,'toConvexCore received')
            raise AssertionError('Any polygon must be cutable')
        oct += 1

        ct = 0
        while hits_e[bestPt,i] > -lpt or circularMinusAbs(i,bestPt,lpt) < 2 or listContain(coini,i) != -1: # Try to find a point that doesn't hit
            i = circularAddInt(i, 1, lpt)
            ct += 1
            if ct >= lpt: 
                print('\033[91mERROR: toConvexCore failed to find cutting point of a polygon (Type II).\033[0m This should never happen. Please report to the author.')
                print('toConvexCore received:\n', sortedPoly)
                print('out:\n',sortedOutsideNorm)
                # plotEdges(sortedPoly,0.5*sortedOutsideNorm,'toConvexCore received')
                raise AssertionError('Any polygon must be cutable')
        cutIdx = i

        if cutIdx > bestPt: 
            aut = bestPt
            yn23 = cutIdx
        else:
            aut = cutIdx
            yn23 = bestPt
        ne = np.array([sortedPoly[yn23-1,1],sortedPoly[aut,0]])
        if np.sum(batchBatchCoincident(np.array([ne]),np.reshape(sortedPoly,(-1,len(ne[0]))),mathematical=False)) <= 4: 
            break
        coini.append(i)
    bufferi = np.concatenate((sortedPoly[aut:yn23],[ne]),axis=0)
    nn = inPlaneNorm(ne[1]-ne[0]) # Get new norm and test which way is it

    # Norm test
    if isOutFlipped(sortedPoly[yn23-1],norms[yn23-1],ne,nn)[0]: 
        nn *= -1
    
    # Concate all sides and keep rolling
    normsOuti = np.concatenate((norms[aut:yn23],[nn]),axis=0)
    bufferj = np.concatenate(([np.flip(ne,axis=0)],sortedPoly[yn23:],sortedPoly[:aut]),axis=0)
    normsOutj = np.concatenate(([-nn],norms[yn23:],norms[:aut]),axis=0)
    bi = bufferi.tolist()
    binn = normsOuti.tolist()
    removeExcessEdges(bi,binn)
    bufferi = np.array(bi)
    normsOuti = np.array(binn)
    bj = bufferj.tolist()
    bjnn = normsOutj.tolist()
    removeExcessEdges(bj,bjnn)
    bufferj = np.array(bj)
    normsOutj = np.array(bjnn)

    if len(bufferi) <= 2 or len(bufferj) <= 2: 
        print('\033[91mtoConvexCore Error: Failed to cut polyon (Type IV).\033[0m')
        print('poly received:\n',sortedPoly)
        print('poly norm received:\n',sortedOutsideNorm)
        # plotEdges(sortedPoly,norms=0.5*sortedOutsideNorm)
        raise AssertionError('Any polygon must be cutable')
    if len(bufferi) >= len(sortedPoly) or len(bufferj) >= len(sortedPoly): 
        print('\033[91mtoConvexCore Error: Failed to cut polyon (Type V).\033[0m')
        print('poly received:\n',sortedPoly)
        print('poly norm received:\n',sortedOutsideNorm)
        # plotEdges(sortedPoly,norms=0.5*sortedOutsideNorm)
        raise AssertionError('Any polygon must be cutable')
    toConvexCore(bufferi,normsOuti,recr=recr)
    toConvexCore(bufferj,normsOutj,recr=recr)

def toConvex(sortedPoly, sortedOutsideNorm): # fck
    """sortedPoly -> a list of convex polygons. Is by itself pretty much an interface. See toConvexCore for the actual algorithm. 
    
    In practice, the convertion results are quite often in triangles already.
    
    Can NOT process polygon with connecting parallel edges. If your input could have such shapes, please avoid them or process with removeExcessEdges() before calling this. 
    """
    if len(sortedPoly) > 500: 
        print('toConvex alert: big poly, may take a while.')
        print('Received # of polygon edges:',len(sortedPoly))
    rtn = []
    toConvexCore(sortedPoly, sortedOutsideNorm, rtn)
    return rtn

def toTrigs(sortedConvexPoly): # fck
    """Converts a sorted, convex polygon (array of edges [nx2xd]) into a list (NOT np.array) of triangles (array of vortices [mx3xd])
    
    Algorithm: Come-on this is a sugery piece of cake given the assumption."""
    if len(sortedConvexPoly) < 3: 
        print('\033[91mtoTrigs error: The input polygon has less than two edges.\033[0m The input is',sortedConvexPoly)
        raise AssertionError('Polygon must have >=3 edges')
    if len(sortedConvexPoly) == 3: 
        return [sortedConvexPoly[:,0]]
    byinh = []
    for i in range(2,len(sortedConvexPoly)): 
        byinh.append(np.array([sortedConvexPoly[0,0], sortedConvexPoly[i-1,0], sortedConvexPoly[i,0]]))
    return byinh

def arrContainBatch(arr1, arr2): # ck
    """Returns if each element in arr2 are contained in arr1 (as a mask where mask[i] = 1 if arr2[i] is contained in arr1, 0 if not). 
    Compares by np.sum(np.abs(arr1[i,...] - arr2[i,...]))
    
    arr1 in [nxaxbxcx...], arr2 in [mxaxbxcx...], return mask in [m]."""
    diffs = np.abs(blurZero(ea1(arr2) - arr1)) # mxnx...
    while len(diffs.shape) > 2: # -> mxn
        diffs = np.sum(diffs, axis=-1)
    diffs = np.min(diffs,axis=-1)
    return 1 - np.sign(diffs)

def tracePoly(poly,pt,eg,outs,outi,lim=trace_limit): # ck # TODO: Change to more efficient version?
    """Helper function. Traces a manifold polygon from poly from pt to direction of eg till back to pt. poly in [nx2xd], pt in [d], eg in [2xd] with one of the points equal to pt
    lim limits the absolute maximum number of edges can be traced. The algorithm will throw an error when lim is exceeded. Does require a outside vector for each edge."""
    assert len(poly) == len(outs)
    yh32 = np.sign(np.linalg.norm(eg-pt,axis=-1))
    pti = np.copy(pt)
    ptn = eg[yh32 == 1]
    traced = [eg]
    tracedn = [outi]
    ct = 0
    while np.linalg.norm(ptn - pti) > coincident_tol:
        if ct > lim: 
            print('\033[91mtracePoly Error: Maximum tracing limiting exceeded.\033[0mYou got a FAT polygon. Please modify trace_limit in config if this is expected.')
            raise AssertionError('Polygon must not have that-many edges')
        ct += 1
        egns = arrContainNorm(poly,ptn) 
        egn = None
        egAngs = np.ones(len(egns)) * np.pi
        for i in range(len(egns)): # Compute angles of all connecting, connect only to the edge with maximum angle (toward-inside) to edge direction
            if not edgeContain(np.array(traced),poly[egns[i,0]]): 
                tst, egAngs[i] = isOutFlipped(traced[-1],tracedn[-1],poly[egns[i,0]],outs[egns[i,0]])
                if tst: 
                    egAngs[i] = np.pi
        i = np.argmin(egAngs)
        if egAngs[i] != np.pi:
            egn = poly[egns[i,0]]
            traced.append(egn)
            tracedn.append(outs[egns[i,0]])

        if egn is None: # Finished
            return np.array(traced), np.array(tracedn)
        # Find next point
        yh32 = np.sign(blurZero(np.linalg.norm(egn-ptn,axis=-1)))
        ptn = egn[yh32 == 1]
    return np.array(traced), np.array(tracedn)

def traceEdgeCore(starti,msk,cntMsk): # ck 
    """Core algorithm: Jump to all that connects to startEg, recursion"""
    if msk[starti//2] == 1: 
        return
    msk[starti//2] = 1
    nextsi = np.squeeze(np.argwhere(cntMsk[starti] == 0),axis=1)
    nextsj = np.squeeze(np.argwhere(cntMsk[otherone(starti)] == 0),axis=1)
    for i in nextsi: 
        nexti = otherone(i)
        traceEdgeCore(nexti,msk,cntMsk)
    for j in nextsj: 
        nextj = otherone(j)
        traceEdgeCore(nextj,msk,cntMsk)
    
def tracePolyAll(poly,startEgi): # ck
    """Helper function. Very simple polygon tracer that finds out which edges in poly are connected to startEg (does NOT care if it should be a 0t point)
    Returns a mask of [len(poly)], where 1s are connected to startEg and 0s are not."""
    msk = np.zeros(len(poly))
    polyPts = np.reshape(poly,(-1,poly.shape[-1]))
    cntMsk = blurZero(np.linalg.norm(polyPts - ea1(polyPts),axis=-1))
    traceEdgeCore(startEgi*2,msk,cntMsk)
    return msk

def breakPolyCore(poly,outs,recr,recn,ignoreLessThan3): # ck
    """Breaks poly with potential 0 thickness points (>2 coincidents at a vortex) into multiple polys without 0t points
    poly in [nx2xd], returns mx[_x2xd] list. Also breaks those are by themselves apart.
    
    Algorithm: Start from a 0t edge, run all the way till back to this vortex, mark that as poly1 and the other as poly2. 
    Recursion on both, till no 0t point can be detected
    """
    if len(poly) <= 2: 
        if ignoreLessThan3: 
            recr.append(poly)
            recn.append(outs)
            return
        print('\033[91mbreakPolyCore ERROR: Illiegal argument: poly has <= 2 edges\033[0m')
        print('poly received:\n',poly)
        print('outs:\n',outs)
        raise AssertionError('Polygon must have >=3 edges')
    if len(poly) <= 3: 
        recr.append(poly)
        recn.append(outs)
        return
    msk = tracePolyAll(poly,0)
    py = poly[msk==0]
    if len(py) > 0: 
        breakPolyCore(py,outs[msk==0],recr,recn,ignoreLessThan3)
    ol = poly[msk==1]
    ns = outs[msk==1]
    pts = np.reshape(ol, (-1,ol.shape[-1])) # 2nxd
    coins = np.sign(-1*np.linalg.norm(blurZero(ea1(pts)-pts),axis=-1))+1 # 2nx2nxd -> 2nx2n
    repeats = np.sum(coins,axis=-1) # -> 2n
    if np.max(repeats) <= 2: 
        recr.append(ol)
        recn.append(ns)
        return 
    repi = np.argmax(repeats)
    rept = pts[repi]
    repe = ol[repi//2]
    outi = ns[repi//2]
    poly1, n1 = tracePoly(ol,rept,repe,ns,outi)
    traced = arrContainBatch(poly1, ol)
    if np.sum(traced) != len(poly1): 
        print('\033[91mbreakPolyCore ERROR: Could not break the polygon (Type I)\033[0m This should never happen. Please report to the author.')
        print('poly received:\n',poly)
        print('outs received:\n',outs)
        print('poly1:\n',poly1)
        raise AssertionError('The traced count must be equal to number of edges traced')
    poly1 = ol[traced==1]
    n1 = ns[traced==1]
    poly2 = ol[traced==0]
    n2 = ns[traced==0]
    if len(poly1) >= len(poly) or len(poly2) >= len(poly): 
        print('\033[91mbreakPolyCore ERROR: Could not break the polygon (Type II)\033[0m This should never happen. Please report to the author.')
        print('poly received:\n',poly)
        print('outs received:\n',outs)
        print('poly1:\n',poly1)
        print('poly2:\n',poly2)
        raise AssertionError('The poly must have been either broken or returned')
    breakPolyCore(poly1,n1,recr,recn,ignoreLessThan3)
    breakPolyCore(poly2,n2,recr,recn,ignoreLessThan3)

def breakPoly(poly, outs, ignoreLessThan3=False): # fck
    """Interface with breakPolyCore. Splits poly into polys from all 0 thickness edges and non-connected spaces. Splits outs by the way.
    If ignoreLessThan3 is set True, when the polygon is split into <3 edge shapes, those results will be returned instead of throwing an error. 
    Any very-legal polygon should not be split into any <3 edge shapes."""
    bud38833357 = []
    ooiiai = []
    breakPolyCore(poly,outs,bud38833357,ooiiai,ignoreLessThan3)
    return bud38833357,ooiiai

def toTrigsOneGo(poly, outs): # fck
    """Shortcut for steps poly(, outsideNorms) -> breakPoly -> sortedPoly -> toConvex -> toTrigs. Returns an numpy array of converted triangles."""
    if len(poly) <= 2: 
        print('toTrigsOneGo ERROR: Received poly with <3 edges.')
        print('Received::\n',poly)
        raise AssertionError('Polygon must have >=3 edges')

    polys, outss = breakPoly(poly, outs)
    cc = []
    for i in range(len(polys)):
        pli = polys[i].tolist() # temporary fix
        nti = outss[i].tolist()
        removeExcessEdges(pli,nti)
        sp, nn = sortPoly(np.array(pli), np.array(nti))
        # plotEdges(sp,nn)
        sold = toConvex(sp, nn)
        for never in sold: 
            cc.append(never)
    autr = []
    for convter in cc: 
        bili = toTrigs(convter)
        for yinhe233 in bili:
            autr.append(yinhe233)
    return np.array(autr)

def inpoly(poly, outside, pt, trigs=None): # fck
    """Returns if pt is within poly (in 2D). If you already have a generated trig mesh of the surface, can use that to save time. 
    In case trigs is filled, both poly and outside are not needed and can be anything."""
    if trigs is None: 
        trigs = toTrigsOneGo(poly,outside)
    for trig in trigs: 
        if inTrig(trig,pt): 
            return True
    return False

def batchInpoly(poly, outside, pts, trigs=None): # fck
    """Batch version of inpoly. Returns a mask where all pt in pts inside the poly is marked 1 and others marked 0.
    poly in [nx2xd] array of edges, outside in [nxd] array of outside norm vectors, pts in [mxd] array of points
    Returning mask in [m]. If you already have a generated trig mesh of the surface, can use that to save time. 
    In case trigs is filled, both poly and outside are not needed and can be anything."""
    if trigs is None:
        trigs = toTrigsOneGo(poly,outside)
    r4913 = np.zeros(len(pts))
    for s13069 in trigs: 
        r4913 += batchInTrig(s13069,pts)
    return np.sign(r4913)

def batchInrect(rect, pts, edgePass=True): # ck
    """Batch compute if all pt in pts are within rect. 
    
    rect in [4xd] array of vortices, pts in [nxd] array of points. 
    
    Returning a in rectangle mask [n], where mask[i] = 1 if pts[i] is inside rect, 0 otherwise
    
    If edgePass is True, those on the rectangle edge will be counted as 1; if False, 0"""
    if len(pts) < 0: 
        return np.array([])
    ed1 = rect[1] - rect[0]
    ed2 = rect[3] - rect[0]
    base = rect[0]
    if np.abs(np.dot(ed1,ed2)) > coincident_tol: # In case of not-that-legal rectangle
        ed2 = rect[2] - rect[0]
    n1 = np.linalg.norm(ed1)
    n2 = np.linalg.norm(ed2)
    ed1 /= dimUp(n1)
    ed2 /= dimUp(n2)
    ptsr = pts - base
    ptsx = np.dot(ptsr,ed1)
    ptsy = np.dot(ptsr,ed2)
    msk = np.ones_like(ptsx)
    if edgePass:
        msk[ptsx < -coincident_tol] = 0
        msk[ptsx > n1+coincident_tol] = 0
        msk[ptsy < -coincident_tol] = 0
        msk[ptsy > n2+coincident_tol] = 0
    else: 
        msk[ptsx < coincident_tol] = 0
        msk[ptsx > n1-coincident_tol] = 0
        msk[ptsy < coincident_tol] = 0
        msk[ptsy > n2-coincident_tol] = 0
    return msk

def toRealPerp(edTar, outi): # fck
    """Helper function. Computes normal direction to edTar that is on the same side as outi. edTar in [2x2], outi in [2]"""
    outj = inPlaneNorm(edTar)
    if np.dot(outj, outi) < 0: 
        outj *= -1
    return outj

def correct89degsAssigner(edges, outs, cntMsk, minCntMsk, e1i, e2i, j0, en1, ed1, ed2): # ck
    """Execution part of correct89degsHelper. Behold of the evil structure..."""
    j0o = otherone(j0)
    edges[e1i, j0o] = edges[e1i,j0] + en1*toRealPerp(ed2, np.sign(0.5-j0)*ed1)
    outs[e1i] = toRealPerp(edges[e1i,1]-edges[e1i,0], outs[e1i])
    next2is = np.squeeze(np.argwhere(minCntMsk[e1i]==0))
    for next2i in next2is: 
        if next2i == e2i or next2i == e1i: 
            continue
        k = np.argwhere(cntMsk[e1i,next2i] == 0)[0]
        edges[next2i, k[1]] = edges[e1i, j0o]
        outs[next2i] = toRealPerp(edges[next2i,1] - edges[next2i,0], outs[next2i])
        correct89degsHelper(edges, outs, cntMsk, minCntMsk, e1i, next2i)

def correct89degsHelper(edges, outs, cntMsk, minCntMsk, e1i, e2i): # ck
    """Helper function. Traces edges from e1i and e2i and correct fake perps until the incoming edge is no longer a fake perp
    Uses a super-evil structure of recursion from sub-helper function."""
    e1 = edges[e1i]
    e2 = edges[e2i]
    ed1 = e1[1] - e1[0]
    ed2 = e2[1] - e2[0]
    en1 = np.linalg.norm(ed1)
    en2 = np.linalg.norm(ed2)
    ed1 /= en1
    ed2 /= en2
    ang = np.abs(np.dot(ed1, ed2))
    if ang <= coincident_tol or ang > perpDotMax: 
        # Stop condition
        return
    j = np.argwhere(cntMsk[e1i,e2i]==0)[0]
    if en1 < en2: 
        correct89degsAssigner(edges, outs, cntMsk, minCntMsk, e1i, e2i, j[0], en1, ed1, ed2)
    else: 
        correct89degsAssigner(edges, outs, cntMsk, minCntMsk, e2i, e1i, j[1], en2, ed2, ed1)

def correct89degs(edges, outs): # ck
    """Correct edges that are close to 90 degrees to 90 degrees to avoid sigularities in further calculations. 
    Input edges in [nx2xd], outs in [nxd]. Outputs corrected edges in [nx2xd], and corresponding outs in [nxd]"""
    
    # Step 1: Check almost-perp angles
    eds = toUnit(edges[:,1] - edges[:,0])
    angTest = np.einsum('nd,md->nm',eds,eds) # nxn
    angTest[angTest <= coincident_tol] = 1 
    angTest[angTest <= perpDotMax] = 0
    angTest[angTest > perpDotMax] = 1
    if len(angTest[angTest==0]) <= 0:
        return edges, outs
    
    # Step 2: Check connection
    pts = np.reshape(edges,[-1,2]) # 2nxd
    cntMsk = np.sign(blurZero(np.linalg.norm(pts-ea1(pts), axis=-1))) # 2nxd - 2nx1xd -n> 2nx2n
    c00 = cntMsk[::2,::2]
    c01 = cntMsk[::2,1::2]
    c10 = cntMsk[1::2,::2]
    c11 = cntMsk[1::2,1::2]
    c0 = np.stack([c00,c10],axis=-1)
    c1 = np.stack([c01,c11],axis=-1)
    cntMsk = np.stack([c0,c1],axis=-1) # nxn x 2x2
    y2h3 = np.tile(dimUp(dimUp(angTest)), (1,1,2,2)) # nxn x 2x2
    cntAngMsk = cntMsk + y2h3 # Points that are both belong to almost-perp edges and connected will be 0. Point coords in 2nx2n, edge coords in nxn of 2x2s 
    passMsk = np.min(cntAngMsk, axis=(-1,-2)) # "which edges are almost-perp and connected"
    if len(passMsk[passMsk==0]) <= 0:
        return edges, outs
    
    # Step 3: For those connected and almost-perp, correct the shorter edge and update the chain.
    minCntMsk = np.min(cntMsk, axis=(-1,-2))
    for i in np.argwhere(passMsk==0): 
        # Note i is [a,b]
        correct89degsHelper(edges, outs, cntMsk, minCntMsk, i[0], i[1])
    return edges, outs

def isNotLeft(pt1, pt2, pt3, norm = None): # fck
    """Helper function. Returns wether pt3-pt2 is to the left of pt2-pt1 (through cross product)
    If above 2D, requests a positive norm direction."""
    if norm is not None: 
        return np.dot(np.cross(pt3-pt2, pt2-pt1),norm) > 0
    return np.cross(pt3-pt2, pt2-pt1) > 0

def toConvexHullHelper(lst): # ck
    """Helper function"""
    i = 0
    while -i < len(lst): 
        while len(lst) >= -(i-3) and isNotLeft(lst[i-3],lst[i-2],lst[i-1]): 
            del lst[i-2]
            if i < 0:
                i += 1
        if len(lst) < -(i-3): 
            break
        i -= 1

def toConvexHull(pts): # ck
    """Convert pts [array of points, nxd] into a convex hull [array of points, mxd, m <= n]. Assert pts in [nx[x,y,...]]
    Uses a slightly modified mixture of Graham's Algorithm and Andrew's Algorithm (Example: https://zhuanlan.zhihu.com/p/340442313)
    This function handles polygon with holes."""
    x_sort = np.argsort(100*pts[:,0]+0.01*pts[:,1])
    ini = pts[x_sort]
    btm = toList1D(ini)
    toConvexHullHelper(btm)
    up = toList1D(ini, reverse=True)
    toConvexHullHelper(up)
    del btm[-1]
    del up[-1]
    return np.concatenate((btm,up),axis=0)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Solver & Finite differecing

def centerDerivMulti(p, consts, func, dp=1e-6): # fck
    """Computes center derivative of func at p. Thinks func has multiple outputs each responding to one of p

    Parameters
    -------
    p : np.array (axbx...xzxm)
        Point of derivation

    consts : tuple
        Constants to be passed to func

    func : function (output axbx...xz)
        Function to take derivative of. Called with func(p+-dp, consts)

    dp : float, optional
        center derivative distance/2. Default to 1e-6

    Returns
    -------
    dpV : np.array (axbx...xzxm)
    Derivative / Gradient of func() at p

    """
    size = p.shape[-1]
    dpV = np.zeros(p.shape)
    for i in range(size):
        d = np.zeros(size)
        d[i] = dp
        posi = func(p+d,consts)
        d[i] = -dp
        negi = func(p+d,consts)
        dpV[...,i] = (posi-negi) / (2*dp)
    return dpV

def solvePGDMulti(arg0, consts, cost, randMin=0.0, randMax=1.0, dCost=1e-6, projection=False, tol=1e-12, itrMax=1000, iniStep=1e-1, showStep=False, totalItr=0): # cnck
    """Performs a projected gradient descent to minimize cost(arg0, consts), where the solver iterates arg0 while passing consts un-modified.

    Parameters
    -------
    arg0 : 1-D np.array (axbx...xzxm)
        Initial guess of the argument. 

    consts : tuple
        Constant values to be passed into cost()

    cost : function (axbx...xz)
        Cost function. Called with cost(arg0, consts)
        MUST have EXACTLY ONE float output PER VECTOR INPUT

    randMin : float
        Minimum value that the function will reset the argment to, when the argment results in a zero derivative point. Default 0

    randMax : float
        Maximum value that the function will reset the argment to, when the argment results in a zero derivative point. Default 1

    dCost : function, optional
        Derivative of cost function, or center derivative distance/2. Called with dCost(arg0, consts) if specified. Defaults to centerDeriv(arg0, consts, dCost=1e-6). 
        Filling in neither float or function could cause an error. 
    
    projection : function, optional
        Projection function to project arg0. Called with projection(arg0, consts) if specified. Defaults to ignore projection
    
    tol : float, optional
        Solver tolerance. Default 1e-12
    
    itrMax : int, optional
        Maximum iteration steps. Default 1000
    
    iniStep : int, optional
        Initial step size. Default 1
    
    showStep : boolean, optional
        Toggle print steps. Default False

    totalItr : Do not fill
        For internal step tracking. Filling-in other than 0 could mess up the step count
        
    Returns
    -------
    arg0 : np.array
        Solved arg0 that minimizes cost() with tolerance specified

    Raises
    -------
    PGD exception : Max iteration reached
        Happens when # of iteration > itrMax. Will print out in console, terminate the function and return current value. 
        When met, would recommend enlarging *iniStep* instead of *itrMax* if you believe the optimization should be possible. 
    
    PGD exception: Derivative is zero
        Happens when np.linalg.norm(dCost()) == 0, in which case the function can not determine where to proceed. 
        Will print out in console, terminate the function and return current value.
    """
    if iniStep <= tol:
        return arg0
    p = arg0
    lastP = np.copy(p)
    i = totalItr
    thisVal = cost(p, consts)
    lastVal = thisVal + 1
    keepingMsk = np.ones_like(p)
    while i<=itrMax and np.sum(keepingMsk) > 0:
        # 
        if isinstance(dCost, float):
            dpV = centerDerivMulti(p, consts, cost, dCost)
        else:
            dpV = dCost(p, consts) # type: ignore
        dpVNorm = np.linalg.norm(dpV,axis=-1)
        if len(dpVNorm==0) != 0: 
            if showStep: 
                print('PGD exception: Derivative is zero! Might hit exactly or more likely having a faulty cost/dCost function. Resetting corresponding values to be a random point.')
            p[dpVNorm==0] = np.random.random(p[dpVNorm==0].shape)*(randMax-randMin) + randMin
            dpV = centerDerivMulti(p, consts, cost, dCost)
            dpVNorm = np.linalg.norm(dpV,axis=-1)
        lastP = np.copy(p)
        p = p - iniStep*dpV/dimUp(dpVNorm)*keepingMsk
        if not isinstance(projection, bool): 
            p = projection(p, consts)
        lastVal = thisVal
        thisVal = cost(p, consts)
        i += 1
        inc = thisVal - lastVal
        keepingMsk[inc>0,:] = 0
        if showStep:
            print('PGD optimization Iteration:', i, '@ step size', "{:.1e}".format(iniStep), ', total step #', i+totalItr, ':\n\tCosts:\n\t', thisVal, '\n\tLast:\n\t',lastVal, '\n\targ: \n\t', p, '\n\tderiv:\n\t', dpV/dimUp(dpVNorm), '\n\tmsk:\n\t', keepingMsk)
    if i > itrMax:
        print('PGD exception: Max iteration reached during step =', "{:.1e}".format(iniStep), 'search. This is usually fine. Come back if you see a weird angle.')
        return p
    return solvePGDMulti(lastP, consts, cost, randMin, randMax, dCost, projection, tol, itrMax, iniStep/10, showStep, i)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Very random global parameters that needs to be computed
testDirs = eqSphereSample(vis_sample_pts)
lightSourceVec = toUnit(np.array(light_source_angle))
