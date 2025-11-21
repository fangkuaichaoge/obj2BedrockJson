from PIL import Image, ImageDraw
from mathHelper import *
# from debuggerFc import *
texture_real_width = texture_width * texture_amplifier

def extractAngle(rects): # ck
    """Extracts norm angles of each rect in rects. rects in [nx4x3]. Returns [nx3] 'angles' that are unit norm vectors.
    Note the extracted norm could be negative of outside."""
    r1 = toUnit(rects[:,1] - rects[:,0])
    r2 = toUnit(rects[:,3] - rects[:,0])
    dotck = blurZero(np.einsum('nd,nd->n',r1,r2))
    r2[dotck != 0] = toUnit(rects[dotck != 0,2] - rects[dotck != 0,0])
    return toUnit(np.cross(r1,r2))
    
def getCenters(rects): # fck
    """Returns center coordinates of each rect in rects. rects in [nx4xd], centers in [nxd]. You may wonder why it's here. Stay toned."""
    return np.average(rects,axis=-2)

def getSizes(rects, scaleToMax = 0.0): # fck
    """Returns sizes of each rect in rects. rects in [nx4xd], sizes in [nx2] with size[:,0] being x length and size[:,1] being y length
    Assumes points in each rect in rects are saved in a loop. If filled scaleToMax > 0, will scale all sizes to have a maximum of this number."""
    v1 = rects[:,1] - rects[:,0]
    v2 = rects[:,3] - rects[:,0]
    dx = np.linalg.norm(v1, axis=-1)
    dy = np.linalg.norm(v2, axis=-1)
    rst = np.stack([dx,dy],axis=-1)
    if scaleToMax <= 0:
        return rst
    return rst / np.max(rst) * scaleToMax

def colorizerHelper(pt, angVec, trigMesh): # ck
    """Helper funtion"""
    tpt = pt + angVec * 2 * coincident_tol
    he233 = shadedInTrigs(trigMesh,tpt,testDirs,raw=True)
    visibleRate = np.min([len(he233[he233<=0])/len(he233)*2,1]) # type:ignore 
    return visibleRate # In theory, a point perfectly outside of the shape should have (actually a bit more than) half of directions un-covered.

def colorizer(angleVecs, rects, trigMesh, baseColor=np.array([255,255,255,255]), minColor=np.array([0,0,0,255])): # ck
    """Generate angle and visibility-based coloring. Color format in RGBA. Those fully covered will be assigned a transparent color (0,0,0,0)
    Note input baseColor is in np.array and is in RGBA ([R, G, B, A]_255_max). Same for minColor. 
    Since the generated norm vector is x cross y, the returned color array contains +z (S in json file) color in colors[...,0,:] and -z (N) in colors[...,1,:]
    
    Input angleVecs in [nx3], rects in [nx4x3], trigMesh in [mx3x3], baseColor in [4_RGBA_255], minColor in [4xRGBA_255].

    Returns colors in [nx2x4]"""
    assert rects.shape[1] == 4
    angEff = (np.dot(angleVecs,lightSourceVec)+1+1e-4)/2
    colorScore = []
    for i in range(len(rects)): 
        ctn = np.average(rects[i],axis=0)
        vis1 = colorizerHelper(ctn, angleVecs[i], trigMesh)
        vis2 = colorizerHelper(ctn, -angleVecs[i], trigMesh)
        samplePts = np.array([])
        if vis1 == 0 and vis2 == 0: 
            reg = ptsToEdges(rects[i]) # 4x2x3
            ns = toUnit(np.array([reg[1,1]-reg[1,0], reg[0,0]-reg[0,1], reg[1,0]-reg[1,1], reg[0,1]-reg[0,0]])) # 4x3
            rndSamp = dimUp(np.random.rand(vis_sample_pts)) * (reg[0,1]-reg[0,0]) + dimUp(np.random.rand(vis_sample_pts)) * (reg[1,1]-reg[1,0])
            samplePts = np.concatenate([np.average(reg,axis=1)+2*coincident_tol*ns, rndSamp],axis=0)
        j = 0
        while vis1 == 0 and vis2 == 0: 
            vis1 = colorizerHelper(samplePts[j], angleVecs[i], trigMesh)
            vis2 = colorizerHelper(samplePts[j], -angleVecs[i], trigMesh)
            j += 1
            if j >= len(samplePts): 
                break
        colorScore.append([angEff[i]*vis1, mirror(angEff[i],0.5)*vis2])
    colorScore = np.array(colorScore)
    colorMsk = np.sign(colorScore)
    minScore = np.min(colorScore)
    maxScore = np.max(colorScore)
    if maxScore - minScore > 0: 
        colorScore = (colorScore - minScore) / (maxScore - minScore)
    colors = dimUp(colorScore) * (baseColor - minColor) + minColor
    colors[colorMsk == 0] = 0
    return np.round(colors)

def writeToPng(file_name, content): # fck
    """Writes to png file. content in (starting_coordinate [nx2], sizes [nx2], color_matrix [nx4])"""
    img = Image.new('RGBA', (texture_real_width, texture_real_width), (0,0,0,0))
    drw = ImageDraw.Draw(img)
    coords, sizes, colors = content
    for i in range(len(coords)): 
        drw.rectangle((coords[i,0],coords[i,1],coords[i,0]+sizes[i,0],coords[i,1]+sizes[i,1]),fill=tuple(colors[i].astype(int))) # two cornors 
    img.save(save_path+'/'+file_name+'.png')

def writeUVpng(file_name, rawUVInfo): # fck
    """Interface to write generated UV info. UV info in (starts [nx2x2], sizes [nx2x2] and colors [nx2x4]). Skippes all with 0 alpha or 0 size."""
    coords, sizes, clrs = rawUVInfo
    tstMsk = sizes[...,0] * sizes[...,1] * clrs[...,-1]
    writeToPng(file_name, (coords[tstMsk!=0], sizes[tstMsk!=0], clrs[tstMsk!=0]))

def fillUVChunk(xDomSize, uvSize): # ck
    """Dummy operator that fills a chunk of uvSize ([x,y]_max, start with [0,0]) with biggest then most rects possible. 
    Returns coords [nx2] as starting coords of each color chunk.
    Super dummy column fill. Looks dumb but is in most cases enoughly working."""
    i = len(xDomSize) - 1
    coords = np.zeros_like(xDomSize)
    while i >= 0: 
        if xDomSize[i][0] == 0 or xDomSize[i][1] == 0: 
            i -= 1
        else: 
            break
    if i <= 0: 
        return coords
    lastX = xDomSize[i,0]
    lasti = i
    i -= 1
    while i >= 0: 
        start = np.array([0,0])
        if xDomSize[i,0] == 0 or xDomSize[i,1] == 0: 
            i -= 1
            continue
        if uvSize[1] - coords[lasti,1] - xDomSize[lasti,1] - xDomSize[i,1] > 0: 
            start = np.array([coords[lasti,0], coords[lasti,1] + xDomSize[lasti,1] + 1])
        else: 
            if uvSize[0] - lastX >= xDomSize[i,0]: 
                start = np.array([lastX+1, 0])
                lastX += xDomSize[i,0]
        coords[i] = start
        lasti = i
        i -= 1
    return coords

def generateUVMap(sizes, colors, uvSize = np.array([texture_real_width, texture_real_width]), uvStart = np.array([0,0]), returnSort = False): # ck
    """Writes colors to UV map. 
    Returns the generated UV map coordinates and sizes of each rectangle, and a color info matrix.
    
    Input sizes in [nx2], colors in [nx2x4] in RGBA, uvSize in [x,y]_max, uvStart in [x,y] coordinate that [0,0] shifts to. Only those with alpha>0 in color will be written to texture file. 
    
    Returns UV mapping info in (starting_coordinate [nx2x2] (north, south), sizes [nx2x2], color_matrix [nx2x4], rotation [nx2]) in their original sequences"""
    rotates = np.zeros(len(sizes))
    rotates[sizes[:,0]<sizes[:,1]] = 90
    szs = dupArr(sizes)
    y3i88n33h35e7233 = np.zeros(len(szs))
    y3i88n33h35e7233[szs[:,0]!=0] = 1
    y3i88n33h35e7233[szs[:,1]!=0] = 1
    szs[y3i88n33h35e7233 == 1] += 2*uv_padding
    rots = dupArr(rotates)
    szs[szs[:,0]<szs[:,1]] = np.flip(szs[szs[:,0]<szs[:,1]], axis=-1)
    xsort = np.argsort(szs[:,0])
    sortedSizes = szs[xsort]
    sortedColors = np.reshape(colors,(-1,4))[xsort]
    sortedRots = rots[xsort]
    sortedSizes[sortedColors[...,-1] == 0] = 0
    starts = fillUVChunk(sortedSizes, uvSize-uvStart)
    y3i88n33h35e7233 = np.zeros(len(sortedSizes))
    y3i88n33h35e7233[sortedSizes[:,0]!=0] = 1
    y3i88n33h35e7233[sortedSizes[:,1]!=0] = 1
    starts[y3i88n33h35e7233 == 1] += uv_padding
    sortedSizes[y3i88n33h35e7233 == 1] -= 2*uv_padding
    revSort = np.argsort(xsort)
    starts += uvStart
    starts = starts[revSort]
    sortedSizes = sortedSizes[revSort]
    sortedColors = sortedColors[revSort]
    sortedRots = sortedRots[revSort]
    starts = np.reshape(starts, (-1,2,2))
    sortedSizes = np.reshape(sortedSizes, (-1,2,2))
    sortedColors = np.reshape(sortedColors, (-1,2,4))
    sortedRots = np.reshape(sortedRots, (-1,2))
    if returnSort: 
        return revSort, starts, sortedSizes, sortedColors, sortedRots
    return starts, sortedSizes, sortedColors, sortedRots

# def inheriteUV(): 
#     """ (WIP) Attempts to keep the old UV mapping. The new faces will be assigned to un-linked areas of the UV texture. Be aware, since the size of old UV is unknown, it could go out of the range.
    
#     Not yet implemented."""
#     # TODO read json to [4xd] rects - uv parameters, compare rects, try to support linear operations like move and scale

def colorizerOneGo(rectss, trigMeshs, uvFileName, uvConfig): # ck
    """External interface to generate uv info and write texture in one-go. 
    
    Input all rects in mx[_x4x3], trigMeshs (for visibility detection) in mx[_x3x3]
    
    Returns uv info in (start_coord mx[_x2], uv_size mx[_x2], uv_rotation mx[_]). Also writes uv texture to configed location."""
    al, go, ri, thm, by, Yin, he233, bili, UI3883, D3357 = uvConfig
    rectCt = 0
    for rects in rectss: 
        rectCt += len(rects)
    if texture_max_width < 0:
        textureMaxWidth = np.floor(texture_real_width/np.sqrt(rectCt*2)) - 2*uv_padding
    elif texture_max_width == 0: 
        textureMaxWidth = np.floor(texture_real_width/np.sqrt(rectCt))
    else: 
        textureMaxWidth = texture_max_width
    sizess = []
    clrss = []
    ctnss = []
    ls = np.zeros(len(rectss),dtype=int)
    for i in range(len(rectss)):
        rects = rectss[i]
        ls[i] = len(rects) + ls[i-1]
        trigMesh = trigMeshs[i]
        angs = extractAngle(rects)
        ctns = getCenters(rects)
        sizes = np.round(getSizes(rects, textureMaxWidth))
        sizes[sizes==0] = 1 # At least visible

        if i < len(base_colors) and len(base_colors[i]) >= 3: 
            if len(base_colors[i]) < 4: 
                baseColor = np.array([base_colors[i][0],base_colors[i][1],base_colors[i][2],255])
            else:
                baseColor = base_colors[i][:4]
        else:
            if len(default_base_color) < 4:
                baseColor = np.array([default_base_color[0], default_base_color[1], default_base_color[2], 255])
            else:
                baseColor = np.array(default_base_color[:4])

        if i < len(min_colors) and len(min_colors[i]) >= 3: 
            if len(min_colors[i]) < 4: 
                minColor = np.array([min_colors[i][0],min_colors[i][1],min_colors[i][2],255])
            else:
                minColor = min_colors[i][:4]
        else:
            if len(default_min_color) < 4:
                minColor = np.array([default_min_color[0],default_min_color[1],default_min_color[2],255])
            else: 
                minColor = np.array(default_min_color[:4])

        colors = colorizer(angs, rects, trigMesh, baseColor, minColor)
        sizess.append(sizes)
        clrss.append(colors)
        ctnss.append(ctns)
    sizess = np.concatenate(sizess, axis=0)
    colorss = np.concatenate(clrss, axis=0)
    uv_coord, uv_size, uv_clr, uv_rot = generateUVMap(sizess, colorss) # type:ignore
    writeUVpng(uvFileName,(uv_coord,uv_size,uv_clr))
    uv_coord = np.split(uv_coord/texture_amplifier, ls[:-1], axis=0)
    uv_size = np.split(uv_size/texture_amplifier, ls[:-1], axis=0)
    uv_rot = np.split(uv_rot, ls[:-1], axis=0)
    return uv_coord, uv_size, uv_rot
