"""Plane class."""
from mathHelper import *

# Helper classes
class plane: # Known problem: getting plane with e1==e2 or e1==e2==0 will result in NaNs. Expecting that to not happen with any reasonable .obj file. 
    def __init__(self, start, e1, e2, norm) -> None:
        self.start = np.array(start) # 3D start point. Can be any point on the plane
        self.e1 = np.array(e1) # 3D plane vector 1
        self.e2 = np.array(e2) # 3D plane vector 2 perpendicular to 1
        self.norm = np.array(norm) # 3D norm of the plane

    def contains(self, pt): # ck
        """Returns if the plane contains this point"""
        return np.abs(np.dot(np.array(pt)-self.start, self.norm)) < coincident_tol
    
    def containsTrig(self, trig):
        """Shortcut for loop"""
        for pt in trig: 
            if not self.contains(pt):
                return False
        return True
    
    def toPlanerCoordMulti(self, pts): # ck
        """Convert an array of pts to planer coords. ASSUMES all pt in pts are on the plane. Those not on the plane could generate weird results."""
        return np.array([np.dot((pts-self.start),self.e1), np.dot((pts-self.start),self.e2)]).T

    def toPlanerCoord(self, pt): # ck
        """Converts 3D pt to planer coordinates. Returns False if not on the plane"""
        if not self.contains(pt): 
            return False
        p = np.array(pt)
        return np.array([np.dot(self.e1,(p-self.start)), np.dot(self.e2,(p-self.start))])
    
    def to3DCoord(self, pt): # ck
        """Converts planer coord to 3D coord"""
        return blurZero(self.start + self.e1 * pt[0] + self.e2 * pt[1])
    
    def to3DCoordMulti(self, pts): 
        """Converts multiple planer coord to 3D coord"""
        return blurZero(self.start + self.e1 * dimUp(pts[...,0]) + self.e2 * dimUp(pts[...,1]))

    def projection(self, pt): # ck
        """Project pt onto this plane, coord in e1 and e2"""
        diff = pt-self.start
        return np.array([np.dot(diff,self.e1), np.dot(diff,self.e2)])
    
    def isParallel(self, dir): # ck
        """Returns if dir direction is parallel with this plane. dir is in [x,y,z]_direction, recommend to use vector with module of 1. [0,0,0] returns True"""
        return np.abs(np.dot(self.norm, np.array(dir))) < coincident_tol
    
    def isPerp(self, dir): # ck
        """Returns if dir is perpendicular to this plane. dir is in [x,y,z]_direction, recommend to use vector with module of 1. [0,0,0] returns True"""
        d = np.array(dir)
        return np.abs(np.dot(self.e1, d)) < coincident_tol and np.abs(np.dot(self.e2, d)) < coincident_tol

    def intersect(self, line): # ck
        """Returns the intersect point between this plane and line as np.array([a*e1,b*e2,c*line[1] (from line[0])]). Returns False if not intersecting. line is saved as [[x,y,z]_start, [x,y,z]_direction].
        If start points are equal, returns np.array([0,0])."""
        if self.contains(line[0]) and self.isParallel(line[1]): 
            return np.append(self.toPlanerCoord(line[0]),0)
        L = np.array([self.e1,self.e2,line[1]])
        b = np.array(line[0])-self.start
        if np.linalg.det(L) == 0: 
            return False
        return np.linalg.inv(L)@b

    def equal(self, p2): 
        """Mathematically the same plane. p1==p2 would be seeking parameter-wise the same plane."""
        return self.contains(p2.start) and (1-coincident_tol < np.abs(np.dot(self.norm,p2.norm)) < 1+coincident_tol)

    def __str__(self): 
        return "Plane content:\n\tstart\t:\t%s \n\te1\t:\t%s \n\te2\t:\t%s \n\tnorm\t:\t%s" % (self.start,self.e1,self.e2,self.norm)
    
    def __eq__(self, p2): 
        return roughEqual([self.start,p2.start],coincident_tol) and roughEqual([self.e1,p2.e1],coincident_tol) and roughEqual([self.e2,p2.e2],coincident_tol)
    
    def hashHelper(self, val, mult):
        return mult * np.sum(val)
    
    def __hash__(self) -> int:
        return self.hashHelper(self.start,256) + self.hashHelper(self.e1,16) + self.hashHelper(self.e2,8) + self.hashHelper(self.norm,1)

