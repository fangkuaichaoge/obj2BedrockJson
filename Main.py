# Setup
import os
from objProcesser import *
from rectFc import *
from toBedrock import *
from shader import *

# Dummy-proof config processing
def unidealParamPrt(paramName, val, should_cn, should_en): # fck
    print('您设置了', paramName, '为', val, '. 请注意这可能不是一个理想的值。一般情况下该变量应该',should_cn)
    print('You have set', paramName, 'to be', val, '. Please be aware this may not be ideal. For most cases, the value should be', should_en)

if not os.path.isfile(file_path): 
    raise AssertionError('obj路径必须有obj. obj file_path must direct an obj file.')
if not os.path.isdir(save_path): 
    os.mkdir(save_path)
if not isinstance(save_name, str): 
    raise AssertionError('save_name必须是str。save_name must be string.')
model_rotation = np.array(model_rotation)
if model_rotation.ndim != 2 or len(model_rotation) != 3 or len(model_rotation[0]) != 3: 
    raise AssertionError('旋转矩阵必须是3x3. Rotation matrix must be 3x3.')
doesRot = np.sum(np.abs(model_rotation - np.eye(3))) > 0
if BBL_per_unit <= 0: 
    raise AssertionError("BBL_per_unit 必须为正 BBL_per_unit must be positive.")
if coincident_tol <= 0: 
    unidealParamPrt('coincident_tol', coincident_tol, '小但>0', 'small but greater than 0')
if vis_sample_pts <= 0: 
    raise AssertionError('vis_sample_pts 必须为正 vis_sample_pts must be positive.')
if not isinstance(fast_intersect,bool): 
    raise AssertionError('fast_intersect必须是T/F. fast_intersect must be T/F.')
if not isinstance(hardcore_hole_test,bool): 
    raise AssertionError('hardcore_hole_test必须是T/F. hardcore_hole_test must be T/F.')
if min_edge_length <= 0: 
    print('您设置了min_edge_length<=0，算法将不会剔除任何小面，输出可能存在大量零碎面。')
    print('You have set min_edge_length to be <= 0. This will disable size-based rectangle culling and could result in shattered rectangle pieces.')
trace_limit = int(trace_limit)
if trace_limit <= 0: 
    raise AssertionError('trace_limit必须为正。trace_limit must be >0.')
if trace_limit < 100: 
    unidealParamPrt('trace_limit', trace_limit, '高到模型摸不到', 'High enough that your model won\'t exceed')
if perpendicular_range == 0: 
    unidealParamPrt('perpendicular_range', perpendicular_range, '小但!=0', 'small but !=0')

if not isinstance(auto_coloring, bool): 
    raise AssertionError('auto_coloring必须是T/F. auto_coloring must be T/F.')
if auto_coloring:
    if not isinstance(uv_file_name, str): 
        raise AssertionError('uv_file_name必须是str。 uv_file_name must be string.')
    for i in range(len(base_colors)): 
        if len(base_colors[i]) < 3 and len(base_colors[i]) != 0: 
            raise AssertionError('base_colors所有填入的参数必须至少3长。 Any filled base_colors must be at least 3 in length.')
    if len(default_base_color) < 3: 
        raise AssertionError('default_base_color必须至少3长。 default_base_color must be at least 3 in length.')
    for i in range(len(min_colors)): 
        if len(min_colors[i]) < 3 and len(min_colors[i]) != 0: 
            raise AssertionError('min_colors所有填入的参数必须至少3长。 Any filled min_colors must be at least 3 in length.')
    if len(default_min_color) < 3: 
        raise AssertionError('default_min_color必须至少3长。 default_min_color must be at least 3 in length.')
    if texture_width*texture_amplifier <= 0: 
        raise AssertionError('UV贴图必须有大小。texture_width*texture_amplifier must be > 0')
    if texture_max_width*texture_amplifier + 2*uv_padding > texture_width*texture_amplifier: 
        print('错误：输入的 texture_max_width*texture_amplifier + 2*uv_padding > texture_width*texture_amplifier')
        print('ERROR: Input texture_max_width*texture_amplifier + 2*uv_padding > texture_width*texture_amplifier')
        raise AssertionError('UV最长面必须能塞进UV贴图。 The longest surface must be able to fit into the UV image.')
    if texture_max_width*texture_amplifier + 2*uv_padding > texture_width*texture_amplifier/2: 
        print('警告：您输入的UV参数组合将产生相当大的UV面，这可能导致UV贴图塞不下所有面')
        print('Warning: The UV texture parameters you set would lead to big UV surfaces, this may cause some surfaces to be excluded from the UV image.')
    if np.linalg.norm(light_source_angle) == 0: 
        raise AssertionError('light_source_angle不可以是[0,0,0]。 light_source_angle must not be [0,0,0].')
    if np.linalg.norm(light_source_angle) < 1e-2: 
        print('警告：您输入的light_source_angle长度很短，这可能因浮点精度问题导致光照偏移。')
        print('Warning: The light_source_angle you entered is very short. This may cause shifts in light source location due to float point issues.')

if save_name == '':
    file_name = getFileName(file_path)
else: 
    file_name = save_name
if uv_file_name == '': 
    uv_file_name = file_name+'_texture'
uvConfig = (uv_file_name, base_colors, default_base_color, min_colors, default_min_color, texture_width, texture_amplifier, texture_max_width, uv_padding, light_source_angle)

# Main 11/11 DONE Convertion part done! Take a minute and celebrate this! :)
ti = tic() 
allTrigs, grps, trigCt = load_obj(file_path)
print('Obj载入完成。开始转换...')
print('Obj file loaded. Starting convertion to rectangular faces...\n')

rects = []
rectCt = 0
trigStart = 0
print('已载入',len(allTrigs),'个体。')
print('Received',len(allTrigs),'bodies in total.\n')
for i in range(len(allTrigs)): 
    buffer = []
    print('---------------------------------------------------------------------')
    print('正在转换第',i+1,'个体，该体有',len(allTrigs[i]),'个三角面。')
    print('Converting body',i+1,'containing',len(allTrigs[i]),'triangles...\n')
    if doesRot:
        trigi = np.reshape(allTrigs[i], (-1,3))
        allTrigs[i] = np.reshape(batchRotate(model_rotation, trigi),(-1,3,3))
    autYh233 = toRects(allTrigs[i],trigCt=trigCt,trigStart=trigStart)
    rectCt += len(autYh233)
    rects.append(autYh233)
    trigStart += len(allTrigs[i])
dt = toc(ti,'Converting geometry',False)
print('---------------------------------------------------------------------')
print('几何转换共花费',dt,'秒')
print('Converting geometry took',dt,'s.\n')

if auto_coloring:
    print('几何转换完成。共得',rectCt,'个长方面。开始着色...')
    print('Convertion complete with',rectCt,'rectangle surfaces in total. Generating coloring...\n')
    t = tic()
    uv_infos = colorizerOneGo(rects, allTrigs, uv_file_name, uvConfig)
    dt = toc(t,'Generating coloring', False)
    print('生成着色共花费',dt,'秒。')
    print('Generating colors took',dt,'s\n')
else: 
    uv_infos = ()

if auto_coloring:
    print('着色完成。开始写入json...')
    print('Coloring done. Writing to json...\n')
else: 
    print('几何转换完成。共得',rectCt,'个长方面。开始写入json...')
    print('Convertion complete with',rectCt,'rectangle surfaces in total. Writing to json...\n')
t = tic()
if auto_coloring: 
    bbjson = toBBJsonUV(rects,grps,file_name,uv_infos)
else:
    bbjson = toBBJson(rects,grps,file_name)
dt = toc(t,'a',False)
print('转换至基岩版json格式花了',dt,'秒。')
print('To bedrock json format convertion completed in',dt,'seconds.\n')
writeToJson(file_name,bbjson)
print('搞定！已保存于','./'+save_path)
print('Done! Check','./'+save_path,'for generated files.\n')
dt = toc(ti,'Whole process',False)
print('全程共花费',dt,'秒。')
print('Whole process took', dt,'s.')
