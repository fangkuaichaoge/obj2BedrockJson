"""Triangle operation functions."""
from plane import *

# Triangle operations
def getPlane(trig): # ck
    """Return the plane the trig is on in [start = trig[0], e1 = trig[1]-trig[0], e2 = perp to e1, norm = perp to plane]"""
    if isinstance(trig, list):
        trig = np.array(trig)
    start = trig[0]
    norm = np.cross(trig[2]-trig[1], trig[1]-trig[0])
    norm = norm / np.linalg.norm(norm)
    l1 = trig[1]-trig[0]
    e1 = l1 / np.linalg.norm(l1)
    e2 = np.cross(norm, e1)
    e2 = e2 / np.linalg.norm(e2)
    return plane(start, e1, e2, norm)

def to2DTrig(trig, p : plane): # ck
    """Returns 2D coords in p of the three trig points"""
    l = []
    for i in range(3): 
        l.append(p.toPlanerCoord(trig[i]))
    return np.array(l)
        
def coveringRectHelper(first, second, third): # ck
    """Generate a rect with edge first-second (or first-third_projection if that falls outside of first-second, t_p-s similarly) and bounds 1-2-3 in 2D"""
    edge1 = second-first
    e3 = third-first
    e4 = third-second
    if np.linalg.norm(edge1) == 0 or np.linalg.norm(e3) == 0: 
        return np.array([[0,0],[0,0],[0,0],[0,0]])
    dir1 = edge1 / np.linalg.norm(edge1)
    dir2 = np.cross(np.append(edge1,0),np.array([0,0,1]))
    dir2 = dir2[:2] / np.linalg.norm(dir2[:2])
    edge2 = np.dot((e3),dir2)*dir2
    e3Proj = np.dot(e3,dir1)
    e4Proj = np.dot(e4,dir1)
    whichEdge = np.argmax([np.linalg.norm(edge1),np.abs(e3Proj),np.abs(e4Proj)])
    if whichEdge == 0: 
        return np.array([first, second, second+edge2, first+edge2])
    elif whichEdge == 1: 
        return np.array([first, e3Proj*dir1, e3Proj*dir1+edge2, first+edge2])
    else: 
        return np.array([second+e4Proj*dir1, second, second+edge2, second+e4Proj*dir1+edge2])

def coveringRect(trig, edgeKept : int, p : plane): # ck
    """Generates a rectangle that bounds the trig with an edge on the edgeKept's edge. 
    edgeKept == 0 : edge on [trig[0], trig[1]]
    edgeKept == 1 : edge on [trig[1], trig[2]]
    edgeKept == 2 : edge on [trig[2], trig[0]]"""
    trig2D = to2DTrig(trig, p)
    rect2D = coveringRectHelper(trig2D[edgeKept],trig2D[edgeKept-2],trig2D[edgeKept-1])
    rect = []
    for i in range(4): 
        rect.append(p.to3DCoord(rect2D[i]))
    return rect