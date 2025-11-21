from plane import *
# from debuggerFc import *
from mathHelper import *

class polygon: 
    """Polygon registering edges. All edges are on the same plane."""
    def __init__(self, p : plane) -> None:
        self.p = p
        self.edges = []
        self.outside = [] # registers which direction is outside for each edge (normal to each edge)
        self.testedHoleEgs = [] # Used to skip repeated hole tests
        self.testedHoleResults = []
        self.testedHoles = []
        self.testedHoleOuts = []

    def debug(self, egs, norms):
        """Debug fast declearation of polygon. Should NEVER be used in the process."""
        self.edges = egs.tolist()
        self.outside = norms.tolist()

    def getVorts(self): # fck
        """Compile self.vorts as np.array([[x,y]_1_edge1, [x,y]_2_edge1, [x,y]_1_edge2 ,[x,y]_2_edge2,...])"""
        if len(self.edges) <= 0:
            return np.array([])
        return np.concatenate(self.edges,axis=0)
    
    def trig2Edges2D(self, trig): # ck
        """Helper function. Convert trig (2D coord) into three edges and three normal directions (2D, outside). Returns 2 variables"""
        edges = np.array([[trig[1],trig[0]],[trig[2],trig[1]],[trig[0],trig[2]]])
        edgeDirs = edges[...,1,:] - edges[...,0,:]
        edgeDirs /= dimUp(np.linalg.norm(edgeDirs,axis=-1))
        norms = inPlaneNorm(edgeDirs)

        mid = np.average(trig,axis=0) # middle point of triangle is garanteed to be inside
        relaMids = mid - trig
        ntst = np.sign(np.einsum('nd,nd->n',relaMids,norms))
        norms *= dimUp(-ntst)

        return edges, norms

    def sharedEdges(self, es): # ck
        """Helper function. Returns indices of self.edges and es that are the same. Is done via mathHelper.shareElement()"""
        if len(self.edges) <= 0: 
            return np.array([])
        edges = np.array(self.edges)
        eq1 = sharedElements(edges, es, True)
        eq2 = sharedElements(edges, np.flip(es,axis=1), True)
        eqs = []
        for i in range(len(eq1)): 
            eqs.append(eq1[i])
        for i in range(len(eq2)): 
            eqs.append(eq2[i])
        
        return np.array(eqs)

    def addTrig(self, trig): # ck
        """Merges trig (in 3D global coord) into this polygon. Any overlapping edges will be removed from self.edges. Returns True upon success, False if trig is not on the same plane as self.
        """
        # Convert to 2D coord
        pts = []
        for pt in trig: 
            pi = self.p.toPlanerCoord(pt)
            if isinstance(pi, bool): 
                return False
            pts.append(pi)
        pts = np.array(pts)

        # Compute overlaps
        es, norms = self.trig2Edges2D(pts)
        overlapping = self.sharedEdges(es)

        # No merging / initialization case
        if len(self.edges) == 0 or len(overlapping) < 1: 
            for e in es:
                self.edges.append(e)
            for n in norms:
                self.outside.append(n)
            return True
        
        # Merging
        overlapping = overlapping[np.argsort(overlapping[:,0])]
        for i in range(len(overlapping)-1,-1,-1): 
            del self.edges[overlapping[i,0]]
            del self.outside[overlapping[i,0]]
        for i in range(len(es)): 
            if listContain(overlapping[:,1], i) == -1: 
                self.edges.append(es[i])
                self.outside.append(norms[i])
        return True

    def addEdgesHelper(self, rect1, pts, edges, outable): # 
        """Helper function. Cuts rect1 into edges at each coincident point in pts. Also returns 'outside' (actually inside) norm
        Outable toggles covering/longest mode"""

        # Prepare rect outside directions
        re = ptsToEdges(rect1)
        rout = inPlaneNorm(edgeDirs(re)) # "out"
        rel = np.linalg.norm([rect1[1]-rect1[0], rect1[3]-rect1[0]],axis=-1)
        mids = (re[:,1]+re[:,0])/2.0
        algBelongs = mids + rout*dimUp(np.array([rel[1]/2, rel[0]/2, rel[1]/2, rel[0]/2]))
        yinhe233OnBilibili = batchInrect(rect1,algBelongs)
        for i in range(len(yinhe233OnBilibili)): 
            if yinhe233OnBilibili[i] == 0: 
                rout[i] *= -1

        # Compute intersections & process pts incase of outable
        if outable:
            its, n = batchIntersect(rect1,re[:,1]-re[:,0],edges)
            n[n>1+coincident_tol] = -1
            pts = np.concatenate([pts,its[n!=-1]],axis=0)
        pts = removeIdentical(pts)

        # Process rect edges
        newes = []
        newns = []
        for i in range(len(re)): 
            ei = re[i]
            coins = pts[batchCoincident(ei, pts, False)==1] 
            if len(coins) <= 0: # Exception: if no hit, go ahead and add
                newes.append(ei)
                newns.append(rout[i])
                continue
            ed = ei[1] - ei[0]
            en = np.linalg.norm(ed)
            ed /= en
            coinDis = np.dot(coins-ei[0],ed)
            coins = coins[np.argsort(coinDis)]

            # Start cutting
            eg = np.array([ei[0],coins[0]])
            if not (edgeContain2(edges, eg) or areOverlapping(eg[0],eg[1])):
                if outable: 
                    if len(hasOutsideEdges(np.array([eg]),np.array([-1*rout[i]]),edges,removeSelf=False))<1: 
                        newes.append(eg)
                        newns.append(rout[i])
                else:
                    newes.append(eg)
                    newns.append(rout[i])
            for j in range(len(coins)-1): 
                eg = np.array([coins[j],coins[j+1]])
                if not (edgeContain2(edges,eg) or areOverlapping(eg[0],eg[1])): 
                    if outable: 
                        if len(hasOutsideEdges(np.array([eg]),np.array([-1*rout[i]]),edges,removeSelf=False))>=1:
                            continue
                    newes.append(eg)
                    newns.append(rout[i])
            eg = np.array([coins[-1],ei[1]])
            if not (edgeContain2(edges, eg) or areOverlapping(eg[0],eg[1])):
                if outable: 
                    if len(hasOutsideEdges(np.array([eg]),np.array([-1*rout[i]]),edges,removeSelf=False))<1: 
                        # ppl('pass',2)
                        newes.append(eg)
                        newns.append(rout[i])
                else:
                    newes.append(eg)
                    newns.append(rout[i])
        return np.array(newes), np.array(newns)

    def toRectHelper(self, edge, out, dis): # fck
        """Rect generation helper. edge [2xd] + out [d] + dis [1] -> rect [4xd]"""
        e2 = edge - out*dis
        return np.array([edge[0],e2[0],e2[1],edge[1]])

    def toRectsBeta(self, rectsi):
        """Processes rectsi (rects_initial) into ready-to-return form. Removes all that are not within trigMesh and fully covered by another rect in rectsi. 
        ASSUMES all points indexed [1,2] are the new points in each rect."""
        rects = []
        for i in range(len(rectsi)): 
            minDis = np.min(rectLengths(rectsi[i]))
            if minDis > min_edge_length:
                rects.append(rectsi[i])
        rects = blurZero(np.array(rects))
        cullMsk = np.zeros(len(rects))
        zzz = np.zeros_like(cullMsk)
        for i in range(len(rects)): # Cull those covered # pck
            if cullMsk[i] > 0: 
                continue
            contained = rectContain(rects,rects[i])
            if np.sum(np.max([contained - cullMsk,zzz],axis=0)) > 1: 
                cullMsk[i] = 1
        rects = np.array(rects[cullMsk<=0])
        rects = np.reshape(rects, (-1,2)) # Final convertion to 3D
        rects = self.p.to3DCoordMulti(rects)
        rects = np.reshape(rects, (-1,4,3))
        return blurZero(rects)

    def holeSampleTest(self, trigMesh, spi, soi, testDirs, debug): 
        """Helper function to avoid copy-paste"""
        # Getting a sample point within the hole -- the actually complicated part
        trgi = np.array([])
        if hardcore_hole_test:
            trgi = toTrigsOneGo(spi,soi) 
        tpts = np.array([[np.average(spi[...,0]),np.average(spi[...,1])]])
        if hardcore_hole_test : # If using hardcore test, grab middle points of its triangles (garanteed inside hole)
            tpts = np.average(trgi,axis=1)
        tpts3 = self.p.to3DCoordMulti(tpts)
        succ = 0
        cnt = 0
        idxs = np.array(np.floor(np.random.rand(vis_sample_pts) * len(tpts3)),dtype=int) # My turn, DRAW!
        for yh in idxs: 
            tpt = tpts3[yh]
            if cnt >= vis_sample_pts: 
                break
            if debug: 
                if trigMesh: 
                    succ += 1
                cnt += 1
                continue
            if shadedInTrigs(trigMesh,tpt,testDirs): 
                succ += 1
            cnt += 1
        if succ >= cnt: 
            return 1, tpts
        return 0, tpts

    def mergePoints(self): 
        """Merges all close points in self.edges (<coincident_tol) into one and recompute norm accordingly. 
        ASSUMES the change is so small that the norm will not be flipped. Is just the simplest for loop.
        Mutates self.edges and self.outside. """
        es = np.array(self.edges)
        pts = np.reshape(es,(-1,2))
        dis = np.linalg.norm(pts - ea1(pts),axis=-1)
        msk = np.zeros(len(pts))
        for i in range(len(pts)): 
            for j in range(len(pts)):
                if msk[j] == 1: 
                    continue
                if dis[i,j] < coincident_tol: 
                    msk[j] = 1
                    self.edges[j//2][j%2] = es[i//2,i%2]
                    e = self.edges[j//2]
                    ot = inPlaneNorm(e[1]-e[0])
                    if np.dot(ot, self.outside[j//2]) < 0:
                        ot *= -1
                    self.outside[j//2] = ot

    def rectTest(self, recti, npts, notHoleEdges, notHoleOuts, trigMesh, testDirs, debug, edgeOutable):
        
        vortPass = 0
        for pt in recti: 
            if len(arrContainNorm(npts,pt)) > 1: 
                vortPass += 1
                continue
            if shadedInTrigs(trigMesh, self.p.to3DCoord(pt), testDirs): 
                vortPass += 1
        if vortPass < 4: 
            return 0.0, notHoleEdges, notHoleOuts, np.zeros(len(notHoleEdges))
        res, ren = self.addEdgesHelper(recti,npts,notHoleEdges,edgeOutable) 
        if len(res) < 1:
            return 1.0, None, None, None # Trivial trivial case. 
        
        # Trace holes by tracePoly from hole edges. 
        allEgs, allOuts = mergeEdges(notHoleEdges, res, notHoleOuts, ren) 
        allIns = -1*allOuts
        allHoles = []
        allOuts = []
        usedMsk = np.zeros(len(res))
        passMsk = np.zeros(len(res))
        invalidMsk = np.zeros(len(res))
        for j in range(len(res)): 
            if usedMsk[j] > 0: 
                invalidMsk[j] = 1
                allHoles.append(np.array([]))
                allOuts.append(np.array([]))
                continue
            re = res[j]
            ckYh233 = edgeContain(np.array(self.testedHoleEgs),re,raw=True) # skip test to avoid repeated hole testing process
            if (not isinstance(ckYh233,bool)) and np.sum(ckYh233) < len(self.testedHoleEgs): 
                idx = np.argwhere(ckYh233==0)[0][0]
                passMsk[j] = self.testedHoleResults[idx]
                allHoles.append(self.testedHoles[idx])
                allOuts.append(self.testedHoleOuts[idx])
                continue
            rn = -1*ren[j]
            ahortu, ye3n2hi3 = tracePoly(allEgs,re[0],re,allIns,rn) 
            allHoles.append(ahortu)
            allOuts.append(-1*ye3n2hi3)
            usedMsk += edgeContainBatch(ahortu,res)

            # Sample points from each hole and test and if no pass continue
            if len(allHoles[j]) < 3: 
                invalidMsk[j] = 1
                self.testedHoleEgs.append(res[j])
                self.testedHoleResults.append(0)
                self.testedHoles.append(allHoles[j])
                self.testedHoleOuts.append(allOuts[j])
                continue
            passMsk[j] = self.holeSampleTest(trigMesh,allHoles[j],-1*allOuts[j],testDirs,debug)[0]
            self.testedHoleEgs.append(res[j])
            self.testedHoleResults.append(passMsk[j])
            self.testedHoles.append(allHoles[j])
            self.testedHoleOuts.append(allOuts[j])
            
        score = 1.0 * np.sum(passMsk + invalidMsk) / len(allHoles)
        return score, allHoles, allOuts, passMsk+invalidMsk

    def toRects(self, trigMesh, debug=False): 
        """ Converts self into an array of rectangles that could cover maximum area of self while not exceeding the boundary of trigMesh (in 3D)
        
        trigMesh in 3D array of triangles (nx3xd). In debug mod, all trigmesh checking are set to trigMesh.

        Converted rectangles in 3D array of rectangles (mx4xd). Note generated rectangles could partially overlap with eachother. 

        Algorithm: 
        Pre-pre-process: Of course, simply return those single rectangle.
        Pre-process: Break polygon and find internal holes
            Create sample points in each internal hole and test if each point is covered in trigMesh
            if all pass: 
                Generates smallest covering rects. For each rectangle, sample rect cornors
                    if all inside trigMesh: 
                        Generate sampler, test all that are within holes formed by the rect and poly edges
                        if still all pass: 
                            Go for it.
                        else:
                            Keep looping for rectangles
                    if all did not pass: 
                        Go to after processing
            after processing (if both above did not pass): 
                Re-create a new polygon given where is free and where is not
                Generate rectangles from each edge that goes to the first intersection with other edges
                Throw out those strictly overlapping
        """
        # Process to remove coincident edges. Must be here because putting in addTrigs could result in in-progress incompatibilities. Should be free to modify self.edges here since it should be called after adding # pck
        self.mergePoints()
        removeExcessEdges(self.edges,self.outside)
        es, ns = correct89degs(np.array(self.edges), np.array(self.outside))
        
        if len(es) == 4: 
            etst = etst = np.reshape(es, (-1,2))
            if np.sum(1-np.sign(np.linalg.norm(etst - ea1(etst),axis=-1))) == len(etst)*2:
                esst, nt = sortPoly(es, ns)
                rl = np.linalg.norm(esst[1,1]-esst[1,0])
                if np.dot(esst[1,1]-esst[1,0], -nt[0]) >= rl - coincident_tol: 
                    rect = self.p.to3DCoordMulti(np.array([esst[0,0],esst[0,1],esst[1,1],esst[2,1]]))
                    return np.array([rect])
            else: 
                print('ERROR: A surface contains 4 edges but is not fully connected. Your file may be broken.')
                raise AssertionError('Rectangles must be manifold')
        
        # Find internal holes
        subpolys, subouts = breakPoly(es,ns)
        holeMsk = np.zeros(len(subpolys))
        holePassMsk = np.zeros(len(subpolys))
        
        holeTpts = []
        if len(subpolys) == 1: 
            notHoleEdges = es
            notHoleOuts = ns
        else:
            for i in range(len(subpolys)): # Known problem: can not handle entirely zig-zag outside edges (with angle <90deg). The shape can't be converted to anything anyway, should be safe to skip handling. 
                covered = hasOutsideEdges(subpolys[i],subouts[i],raw=True)
                covered = np.sum(np.sign(covered))
                if covered >= len(subpolys[i]): 
                    holeMsk[i] = 1
                    
                    # If is a hole, generate sample point(s) and test
                    spi = subpolys[i]
                    soi = subouts[i]
                    tstAndTpt = self.holeSampleTest(trigMesh, spi, soi, testDirs, debug)
                    holePassMsk[i] = tstAndTpt[0]
                    if tstAndTpt[0] == 0:
                        holeTpts.append(tstAndTpt[1])
            notHoleEdges = listConcateWithCond(subpolys,holeMsk,0)
            notHoleOuts = listConcateWithCond(subouts,holeMsk,0)

        npts = np.reshape(notHoleEdges, (-1,2))
        rought1 = True
        rectEgs = np.array([])
        rectNs = np.array([])
        bestPassRate = -1
        bestAllHoles = []
        bestAllOuts = []
        bestPassMsk = np.array([])

        if np.sum(holePassMsk) == np.sum(holeMsk):
            
            # Attempt to generate rectangles
            rects = smallestCoveringRect(es,returnAll=True) # Need to try for all possible covering rects.
            
            # Skip those exactly equal
            skipRect = np.zeros(len(rects)) 
            zzz = np.zeros(len(rects))
            for i in range(len(rects)): 
                if np.sum(np.max([rectContainStrict(rects,rects[i]) - skipRect,zzz],axis=0)) > 1: 
                    skipRect[i] = 1
            
            # Compute if possible to cover by a single rect
            for i in range(len(rects)): 
                recti = blurZero(rects[i])
                if skipRect[i] == 1: 
                    continue
                if len(holeTpts) > 0 and np.sum(batchInrect(recti, holeTpts, edgePass=False)) > 0:
                    raise AssertionError('This line should not be executed in anyway')
                score, allHoles, allOuts, passMsk = self.rectTest(recti, npts, notHoleEdges, notHoleOuts, trigMesh, testDirs, debug, False)
                if score >= 0: 
                    rought1 = False
                if allHoles is None or allOuts is None or passMsk is None or score >= 1.0: 
                    return np.array([self.p.to3DCoordMulti(recti)]) # 'Trivial' case. For real now. 
                if score > bestPassRate: 
                    bestPassRate = score
                    bestAllHoles = allHoles.copy()
                    bestAllOuts = allOuts.copy()
                    bestPassMsk = passMsk.copy()
        
        # For the best rectangle, log which ones are pass. Cover maximum of holes for each continous pass if possible
        if bestPassRate <= 0: 
            rought1 = True
        else:
            assert len(bestAllHoles) >= 1
            needRmv = False
            rectEgs = []
            rectNs = []
            usedOld = np.zeros(len(es))
            for i in range(len(bestPassMsk)): 
                wasOld = edgeContainBatch(es,bestAllHoles[i])
                usedOld += edgeContainBatch(bestAllHoles[i],es)
                
                # If no pass, add original edges. If pass, add new edges (and make sure it run through remove excess if any of this happens)
                if len(bestAllHoles[i]) < 1: 
                    continue
                if bestPassMsk[i] == 0: 
                    le = bestAllHoles[i][wasOld==1]
                    ln = bestAllOuts[i][wasOld==1]
                    for j in range(len(le)):
                        rectEgs.append(le[j])
                        rectNs.append(ln[j])
                else: 
                    le = bestAllHoles[i][wasOld==0]
                    ln = bestAllOuts[i][wasOld==0]
                    for j in range(len(le)):
                        rectEgs.append(le[j])
                        rectNs.append(-ln[j])
                    needRmv = True
            ee = es[usedOld==0]
            nn = ns[usedOld==0]
            for i in range(len(ee)): 
                rectEgs.append(ee[i])
                rectNs.append(nn[i])
            if needRmv: 
                removeExcessEdges(rectEgs,rectNs)
            rectEgs = np.array(rectEgs)
            rectNs = np.array(rectNs)
        newEs = []
        newOuts = []
        newENH = []
        newONH = []
        newEH = []
        newOH = []
        if rought1:
            for i in range(len(subpolys)): 
                if holeMsk[i] == 0: 
                    newENH.append(subpolys[i])
                    newONH.append(subouts[i])
                else: 
                    if holePassMsk[i] == 0: 
                        newEH.append(subpolys[i])
                        newOH.append(subouts[i])
            newENH = np.concatenate(newENH,axis=0)
            newONH = np.concatenate(newONH,axis=0)
            if len(newEH) > 0:
                newEH = np.concatenate(newEH,axis=0)
                newOH = np.concatenate(newOH,axis=0)
                newEs = np.concatenate([newENH,newEH],axis=0)
                newOuts = np.concatenate([newONH,newOH],axis=0)
            else:
                newEs = newENH
                newOuts = newONH
        else: 
            newEs = rectEgs
            newOuts = rectNs
            newENH = rectEgs
            newONH = rectNs

        # Longest run to fully construct new edges 
        # DONE think about the rect-spike case. Maybe a longest run isn't what's needed... But rather running from longest to shortest
        # Also sort the edges so the generation starts from the longest edge and skippes all that being contained by the rect
        newVs = np.reshape(newEs,(-1,2))
        dupOuts = dupArr(newOuts)
        its, n = batchIntersect(newVs, -1*dupOuts, newEs) 
        vortPassMsk = ptsInsideTheLane(newEs, -1*newOuts, newVs)
        vortPassMsk[vortPassMsk==-1e-12] = 0
        vortPassMsk = transpose2D(vortPassMsk)
        vortDis = blurZero(np.einsum('nmd,nd->nm',newVs-ea1(newEs[:,0]),-1*newOuts))*vortPassMsk
        
        # Merge 2 points into 1 edge
        n = np.reshape(n,(len(n)//2,-1))
        dists = np.sort(np.concatenate([n,vortDis],axis=1),axis=1)
        rects = []
        if len(holeTpts) > 0:
            holeTpts = np.concatenate(holeTpts, axis=0)
        for i in range(len(newEs)): 
            lastDis = -114514
            for j in range(len(dists[0])-1, -1, -1): 
                if lastDis == dists[i,j]: 
                    continue
                if dists[i,j] <= 0: 
                    break
                lastDis = dists[i,j]
                recti = self.toRectHelper(newEs[i], newOuts[i], dists[i,j])
                if len(newEH) > 0 and np.sum(batchInrect(recti, holeTpts, edgePass=False)) > 0: 
                    continue
                score, allHoles, allOuts, passMsk = self.rectTest(recti, newVs, newENH, newONH, trigMesh, testDirs, debug, True)

                if allHoles is None or allOuts is None or passMsk is None or score >= 1: 
                    rects.append(recti)
                    break
        return self.toRectsBeta(rects)

    def __str__(self): 
        return "Polygon content:\n\tedges\t:\t%s \n\toutside\t:\t%s \n\t%s" % (self.edges,self.outside,self.p)
