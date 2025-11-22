"""Import and process .obj file."""
from mathHelper import *

# Obj file data processing
def anyShapeToTrigs(vtxs): # fck 
    if len(vtxs) == 3: 
        return [vtxs]
    if len (vtxs) < 3: 
        print('anyShapeToTrigs: Non-complete face detected! This may have an impact on the outcome.')
        return [vtxs]
    rtn = [] 
    for i in range(2,len(vtxs)): 
        rtn.append([vtxs[0],vtxs[i-1],vtxs[i]])
    return rtn

def toTrigFaces(all_faces): # fck: has external realization anyway
    """Converts faces of any shapes to triangles-only. Leaves all triangles untouched"""
    new_faces = []
    for faces in all_faces: 
        fs = []
        for face in faces: 
            fs.append(f for f in anyShapeToTrigs(face))
        new_faces.append(np.array(fs))
    return new_faces

def load_obj(obj_path): # ck
    """Extracts triangular surfaces from obj file. Returns nx[_x3x3] 3D triangular surfaces of each body, [n] name of bodies and scaler count of trig surfaces."""
    # Extraction code adapted from https://github.com/Cracko298/obj2mcpe/blob/main/obj2mcpe.py 
    # BIG THANKS for your work @Cracko298! 
    all_faces = []
    faces = []
    grps = []
    trigCt = 0
    with open(obj_path, "r") as obj_file: 
        vertices = []
        for line in obj_file: 
            if line.startswith('g ') or line.startswith('s ') or line.startswith('mg ') or line.startswith('o '): 
                # Body -> Group
                if len(faces) > 0: 
                    all_faces.append(np.array(faces.copy()))
                    faces = []
                _, n = line.split()
                grps.append(n)
            if line.startswith('v '): 
                # vertices
                _, x, y, z = line.split() 
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '): 
                # face
                face_indices = [int(index.split("/")[0]) - 1 for index in line.split()[1:]]
                if len(face_indices) > 3: 
                    print('暂不支持多边形面obj文件。请先转换为三角形面obj :/')
                    print('Polygon face obj models are not yet supported. Please convert them to triangular face objs :/')
                    assert len(face_indices) <= 3
                faces.append([vertices[i] for i in face_indices])
                trigCt += 1
        if len(faces) > 0: 
            all_faces.append(np.array(faces))
    return all_faces, grps, trigCt