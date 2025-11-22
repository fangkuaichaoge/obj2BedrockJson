# from debuggerFc import *
from mathHelper import *
import json

def rectsToCubes(rects, uv_mapping = None): 
    """Converts vortice-based rectangles to BB format cubes. Generates 1 cube per rect face.
    rects in [nx4x3]. Assumes points in each rect in rects are saved in a loop. 
    Asserts len(uv_mapping) == len(rects), and uv_mapping in (uv_coords [nx2x2], uv_sizes [nx2x2], uv_rotations [nx2])
    """
    uv_coords, uv_sizes, uv_rots = (np.array([]),np.array([]),np.array([])) # single body func
    if uv_mapping != None: 
        uv_coords, uv_sizes, uv_rots = uv_mapping
    v1 = rects[:,1] - rects[:,0]
    v2 = rects[:,3] - rects[:,0]
    v3 = np.cross(v1,v2,axis=-1)
    dx = np.linalg.norm(v1, axis=-1)
    dy = np.linalg.norm(v2, axis=-1)
    v1 /= dimUp(dx)
    v2 /= dimUp(dy)
    v3 /= dimUp(np.linalg.norm(v3,axis=-1))
    vs = transpose2D(np.array([v1,v2,v3]),0,1)
    R = transpose2D(vs,-1,-2)
    eulers = R2Euler(R)
    angs = eulers
    locked = True
    angLocks = []
    angLocks1 = np.argwhere(np.abs(eulers[:,1])<np.pi/2+5*np.pi/180)
    angLocks2 = np.argwhere(np.abs(eulers[:,1])>np.pi/2-5*np.pi/180)
    if len(angLocks1) == 0 or len(angLocks2) == 0:
        locked = False
    if locked:
        angLocks1 = np.concatenate(angLocks1,axis=0)
        angLocks2 = np.concatenate(angLocks2,axis=0)
        angLocks = sharedElements(angLocks1,angLocks2)
        angs[angLocks] = R2EulerNum(R[angLocks],iniGuess=eulers[angLocks])

    angs *= 180.0/np.pi
    angs[:,0] = - angs[:,0] # BB just want these two angles reversed for no reason
    angs[:,1] = - angs[:,1]
    pos = rects[:,0]
    pos[:,0] = -pos[:,0] # BB decides to import all x as -x
    size = np.zeros([len(dx),3])
    size[:,0] = dx
    ori = pos - size # ...and the origin is just saved with a dx difference to position
    size[:,1] = dy
    rtn = []
    
    pos *= BBL_per_unit
    size *= BBL_per_unit
    ori *= BBL_per_unit

    for i in range(len(angs)): 
        if auto_coloring:
            if np.sum(np.abs(uv_sizes[i,0])) > 0 and np.sum(np.abs(uv_sizes[i,1])) > 0: 
                rtn.append({
                    "origin":ori[i].tolist(), "size":size[i].tolist(), "pivot":pos[i].tolist(), "rotation":angs[i].tolist(), 
                    "uv":{
                        "south":{"uv":uv_coords[i,0].tolist(), "uv_size":uv_sizes[i,0].tolist(), "uv_rotation":uv_rots[i,0]},
                        "north":{"uv":uv_coords[i,1].tolist(), "uv_size":uv_sizes[i,1].tolist(), "uv_rotation":uv_rots[i,1]}
                        }})
                continue
            elif np.sum(np.abs(uv_sizes[i,0])) > 0: 
                rtn.append({
                    "origin":ori[i].tolist(), "size":size[i].tolist(), "pivot":pos[i].tolist(), "rotation":angs[i].tolist(), 
                    "uv":{
                        "south":{"uv":uv_coords[i,0].tolist(), "uv_size":uv_sizes[i,0].tolist(), "uv_rotation":uv_rots[i,0]}
                        }})
                continue
            elif np.sum(np.abs(uv_sizes[i,1])) > 0:
                rtn.append({
                    "origin":ori[i].tolist(), "size":size[i].tolist(), "pivot":pos[i].tolist(), "rotation":angs[i].tolist(), 
                    "uv":{
                        "north":{"uv":uv_coords[i,1].tolist(), "uv_size":uv_sizes[i,1].tolist(), "uv_rotation":uv_rots[i,1]}
                        }})
                continue
        rtn.append({"origin":ori[i].tolist(), "size":size[i].tolist(), "pivot":pos[i].tolist(), "rotation":angs[i].tolist()})
    return rtn

def boneFormater(bone_name, cubes, pivot = [0,0,0], parent = None):
    """Writes a bone, given formatted cubes."""
    bone = {"name":bone_name, "pivot":pivot}
    if parent != None: 
        bone.update({"parent":parent})
    bone.update({"cubes":cubes})
    return bone

def toBBJson(all_rects, grps, file_name): 
    """Convert vortice-based rects to json format used in BlockBench Bedrock model. 
    Each grps[i] is used to name the bone of all_rects[i]
    If grps is shorter than all_rects, automatically create default group names
    Note all groups are default to pivot at [0,0,0], have position at [0,0,0] and rotation of [0,0,0].
    Can accept appendix already in BB format to append at the end of all bones. Usually used to append specific bone infomations."""
    # 确保grps列表长度与all_rects匹配
    if len(grps) < len(all_rects):
        print(f"Warning: grps list length ({len(grps)}) is less than all_rects length ({len(all_rects)}). Creating default group names.")
        # 为缺少的组创建默认名称
        default_grps = [f"bone_{i}" for i in range(len(all_rects) - len(grps))]
        grps = grps + default_grps
    elif len(grps) > len(all_rects):
        print(f"Warning: grps list length ({len(grps)}) exceeds all_rects length ({len(all_rects)}). Truncating grps list.")
        grps = grps[:len(all_rects)]
    bbjson = {}
    stuff = []
    for i in range(len(all_rects)): 
        if len(all_rects[i]) <= 0:
            print('Warning: toBBJson() received a body with 0 faces. This may result in a missing part in the converted model.')
            continue
        cubes = rectsToCubes(all_rects[i])
        stuff.append(boneFormater(grps[i],cubes))
    crust = [{"description":{
        "identifier" : "geometry."+file_name
    }, 
              "bones":stuff}]
    bbjson.update({"format_version" : "1.12.0"})
    bbjson.update({"minecraft:geometry" : crust})
    return bbjson

def toBBJsonUV(all_rects, grps, file_name, uvInfo): 
    """Same as toBBJson except the UV version."""
    assert len(all_rects) == len(grps)
    coords, sizes, rots = uvInfo
    bbjson = {}
    stuff = []
    for i in range(len(all_rects)): 
        if len(all_rects[i]) <= 0:
            print('Warning: toBBJson() received a body with 0 faces. This may result in a missing part in the converted model.')
            continue
        cubes = rectsToCubes(all_rects[i], (coords[i], sizes[i], rots[i]))
        stuff.append(boneFormater(grps[i],cubes))
    crust = [{"description":{
        "identifier" : "geometry."+file_name,
        "texture_width": texture_width/texture_amplifier,
        "texture_height": texture_width/texture_amplifier
    }, 
              "bones":stuff}]
    bbjson.update({"format_version" : "1.12.0"})
    bbjson.update({"minecraft:geometry" : crust})
    return bbjson

# WIP
# def appendBones(bbJson, grp_names, grp_pivots, grp_rotates): 
#     """Append empty groups at the end of the bbJson list. 
#     Each empty group is specified with grp_names[i], grp_pivots[i] and grp_rotates[i]
#     Asserts len(grp_names) == len(grp_pivots)"""
#     assert len(grp_names) == len(grp_pivots)
#     # TODO

def writeToJson(file_name, content): # ck
    import os
    # 确保save_path目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 使用os.path.join正确处理路径
    path = os.path.join(save_path, file_name + '.json')
    with open(path, 'w', encoding='utf-8') as f: 
        json.dump(content, f, indent=4, ensure_ascii=False)
