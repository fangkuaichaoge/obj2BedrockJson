from trigOp import *
# from debuggerFc import *
from polygon import *

def toRects(trigs, trigCt=0, trigStart=0): 
    """Overall convertion wrapper. Enter trigCt > 0 to enable progress print out"""
    polys = []
    ti = tic()
    t0 = ti
    yh233 = 0
    for trig in trigs: 
        yh233 += 1
        if len(polys) <= 0: 
            polys.append(polygon(getPlane(trig)))
            polys[-1].addTrig(trig)
            continue
        auty23 = False
        for poly in polys: 
            if poly.addTrig(trig):
                auty23 = True 
                break
        if not auty23: 
            polys.append(polygon(getPlane(trig)))
            polys[-1].addTrig(trig)
        dt = toc(t0,'',prt=False)
        if trigCt > 0 and isTensPercent(yh233+trigStart,trigCt) and dt > 5: 
            print('载入三角面已完成%s%%. 距离开始已使用%s毫秒。' % (round(yh233/trigCt,3)*100,round(toc(ti,'',prt=False),3)*1000))
            print('Adding trigs is %s%% done. %sms used since start adding trigs.' % (round(yh233/trigCt,3)*100,round(toc(ti,'',prt=False),3)*1000))
            t0 = tic()
    dt = toc(ti, 'Add trigs for this body',False)
    print('该体载入三角面花费了',dt,'秒')
    print('Add trigs for this body took',dt,'s.\n')
    print('开始转换为长方面，多半要跑一会儿...')
    print('Start converting to rectangles... This may take a while...\n')
    ti = tic()
    t0 = ti
    allRects = []
    i = 0
    for poly in polys: 
        rst = poly.toRects(trigs)
        if len(rst) > 0:
            for rc in rst:
                allRects.append(rc)
        i += 1
        dt = toc(t0,'',prt=False)
        if trigCt > 0 and isTensPercent(i,len(polys)) and dt > 5: 
            print('转换该体已完成%s%%. 距离开始已使用%s秒。' % (round(i/len(polys),3)*100,round(toc(ti,'',prt=False),3)))
            print('Converting to rectangles (for this body) is %s%% done. %ss used since start converting to rectangles.' % (round(i/len(polys),3)*100,round(toc(ti,'',prt=False),3)))
            t0 = tic()
    dt = toc(ti, 'Convert this body to rects', False)
    print('该体转换为长方面花费了',dt,'秒')
    print('Convert this body to rects took',dt,'s.\n')
    return np.array(allRects)
    
