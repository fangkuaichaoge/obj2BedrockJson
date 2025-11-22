"""Global parameters. Time complexity of this converter is ~O(n^2) in adding triangles, ~O(n^3) in converting to rectangles, with respect to # of triangle faces in your model."""
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 模型转换参数配置 Converter config ###

# obj文件相对路径（仅支持单文件） 
# 小贴士：本作一般耗时最长的转换部分对于模型单体面数复杂度大约为O(n^3)，如果转换时间太长，可以试试切割模型降低单体面数，代价是会多出一些衔接面。
# Relative file path to obj (single file only)
# Pro-tip: The usually-the-most-time-comsuming convertion part of this algorithm is roughly O(n^3) wrt # of triangles in each body. 
# If it takes a long time to convert, splitting your model into more bodies each with less surface count might help, with the cost of having more faces at the split after convertion.
# 时间复杂度影响未标注默认无/微量。
# Time complexity effect default none/minimum for any unlabeled. 
file_path = 'obj/tst.obj'

# json输出相对路径 
# json save path (relative)
save_path = 'json'

# 模型保存名称（无需带.json）。如果留空('')，默认继承原模型名称。
# File save name (no .json). If left blank (''), defaults to inherite the original file name. 
save_name = ''

# 模型造错轴了？模型旋转矩阵，即某种意义上转换时模型旋转角度。默认旋转中心在原点。
# 向量小知识：矩阵竖轴(column)即为对应单元向量转换后的向量。例如[[0,0,1],[1,0,0],[0,1,0]]会将x轴[1,0,0]投影到y轴[0,1,0]，y轴[0,1,0]到z轴[0,0,1]，z[0,0,1]到x[1,0,0]
# 如不需要，请输入identity matrix [[1,0,0],[0,1,0],[0,0,1]]
# Built model on the wrong direction? Rotation/flip matrix. 'Angles' to rotate the entire model while converting. Rotates around [0,0,0].
# Pro-tip: each column is what each axis becomes. e.g. [[0,0,1],[1,0,0],[0,1,0]] assigns x[1,0,0]->y[0,1,0],y[0,1,0]->z[0,0,1],z[0,0,1]->x[1,0,0]. 
# If no rotation, please fill identity [[1,0,0],[0,1,0],[0,0,1]]. 
# 作者废话：xyz欧拉角也许看起来更直观，但实际使用时只要需要转两轴以上它的数值就会变得很怪，还不如手写矩阵来得方便（
# Side note: Euler angles would seem more intuitive but trust me the numbers'll become super weird once you need to rotate around 2 axis or more. 
model_rotation = [[0,1,0],[0,0,1],[1,0,0]] 

# BB单位转换倍率。obj模型的每1单位长度会被转换为这个数字的BB单位长度。（例：如使用0.048BB长度每毫米，可填入0.048来转换毫米制模型）
# 请注意obj文件不一定会使用你在CAD里指定的单位。
# Time complexity effect: None. BlockBench length unit per unit in the model file (e.g. 0.048 length in blockbench per millimeter in mm-based model -> BBL_per_unit = 0.048). 
# Be aware objs may be saved in a different unit than the unit you used inside the CAD. 
BBL_per_unit = 0.48 # >0. 

# 点容差。距离小于该数字点线会被认为相交，或...的两点会被认为是同一个点。
# 请不要填得太大或太小，容易发生匪夷所思的现象。该参数大多数情况下只是用于处理浮点精度问题。
# Tolerance below which a point is considered coincident with a line, or two points are considered the same. 
# Too large or too small of this can cause weird results and errors. Is mostly used to deal with float point issues.
coincident_tol = 1e-4 # >=0. 

# 视线判断点密度。越高可见判定越准确，但计算速度越慢。
# 用于剔除多余面的可见判断算法。等于一条90°的弧线上存在的判定点数字，球面总共判定方向大约是16/pi*这个^2
# 同时也控制部分2D算法内45°弧线上或随机部分有多少采样点
# The higher the better results, but also much higher computational cost. 
# Controls sample density for visibility algorithm used in excess rectangle culling. 
# Is exactly how many points sampled along a 90 degree linear arc, amount of total sample points is roughly 16/pi*this^2. 
# Also controls how many samples are drawn along a 45 deg arc / in the random sample part in some 2D computations.
# 时间复杂度影响：O(n^2)
# Time complexity effect: O(n^2). 
vis_sample_pts = 4 # >= 1

# 是否启用快速交点判断。用于视线判断的核心方法。如关闭(False)会“更正确”地轮询所有xyz的排布来排除/0问题，如开启(True)理论上应该原地爆炸但实际上基本都能用x
# If disabled, will 'more correctly' compute all 6 arangements of xyz when solving for trig mesh intersections; 
# if enabled, will just go for 1, which should blow up in theory but usually works enoughly well. 
# 时间复杂度影响：O(1)，关闭时大约耗时+20%
# Time complexity effect: O(1) (+~20% when set False).
fast_intersect = True # T/F. 

# 设置为True将确保洞的可见性采样位于洞内（信不信由你这个算法究极复杂...）；False时则仅采样洞周边点的算术平均点。
# 目前尚有点bug，设置为True时可能会出现匪夷所思的报错。如果您在转换中遇到报错，请先关闭该设置。
# Controls weither to make sure the hole testing points are inside the hole (believe or not this is really complicated...). 
# If false, the algorithm will check only the coordinate average of all vortices around the hole. 
# Is still buggy, may throw weird errors when set True. If you see errors being thrown, please try setting this to False before reporting to the author. 
# 时间复杂度影响：True +~50%
# Time complexity effect: True +~50%
hardcore_hole_test = True # T/F. 

# 面剔除界限（原模型单位）。生成后任意一边小于该长度的面将不会被保存。任意小于该尺寸的孔洞将默认可以填充，以此减少最终面数。
# Minimum rectangle face edge length in the model's unit. 
# All rects that have an edge below this value will not show up in the final converted model, 
# and all holes smaller than this size will not be checked and probably filled by the surface rectangle to reduce face count
min_edge_length = 1e-2 # >=0.

# 最大单面可追踪边数。如果有某一面的边数超过该值可能会报错。
# 如果您看见“tracePoly Error: Maximum tracing limiting exceeded”报错，请先尝试增加这个数字。
# 但如果您需要将这个数改得很高才能运行，这边更建议尝试简化下模型。
# Absolute maximum number of edges the algorithm can trace along a surface. Setting too low could result in errors. 
# If you see "tracePoly Error: Maximum tracing limiting exceeded", please try increasing this before reporting to the author. 
# If you need to set this super high for the algorithm to run, you may want to try simplify your model first. 
# 时间复杂度影响：理想情况下无
# Time complexity effect: Ideally none. 
trace_limit = 514 # >0. 

# 垂直角度容差。单面内夹角在90°±该数字的两边会被修正为90°来避免奇点问题。基本上是处理CAD保存精度用的。
# 填的太高可能会扭曲面的形状。（单位度）
# Edge angles 90deg+-this will be corrected to 90 degrees to avoid sigularity issues. 
# This is mostly caused by accuracy loss when saving obj files so a small number is desired. 
# setting this too high could cause distortion in face shapes. 
perpendicular_range = 1 # deg. >0 (<0 would work as if abs).


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 自动上色配置 Auto UV config ###

# 是否启用自动上色（False时以下参数皆不会生效）
# Toggles should the algorithm generate auto coloring. (When set False all parameters below will be ineffective)
# 时间复杂度影响：O(n^3)（True时大约+30%时间）
# Time complexity effect: O(n^3) (+~30% time when True). 
auto_coloring = True # T/F.

# UV材质文件保存名称（路径默认与json相同）
# 如留空('')，则默认使用文件名+_texture.png（注：填写时不需要.png）
# UV file save name (directory is same as json). 
# Default to save_name + _texture.png when left blank ('') (note no .png when filling in a new name)
uv_file_name = '' 

# 自动上色中“最亮”*的颜色的RGBA/RGB颜色代码（支持每个体单独设置）
# 对于未填充或为空集[]的体，默认使用default_base_color
# *实际定义为每个体中最面向光源且可见度最高的面的颜色
# 其他面将依据角度和可见性被赋予该值到min_colors中间的值
# 任何只填入三个数的部分将被默认使用RGB并追加透明度255（即不透明）
# RGBA/RGB values of each body's 'brightest' color ([[R,G,B,Alpha]_255 or [R,G,B]_255,...]) 
# i.e. the color that shows up when the surface is facing the light source and is highly visible. 
# Surfaces on this body will get intermediate colors between this and min_colors based on their angles and visibilities. 
# If fill in only 3 values, will interprete as RGB and use alpha of 255. (opaque)
# 请注意尽管该自动生成算法支持透明度，您想使用的渲染引擎不一定支持。请在设计含透明度的颜色前仔细检查支持度。
# Note though this generator supports transparency (alpha), your target rendering engine may not. 
# Please double check before applying alpha < 255. 
base_colors = [] 

# base_colors未填充部分的默认颜色，RGB/RGBA
# Default color for any unfilled or [] in base_colors. RGB/RGBA
default_base_color = [255,255,255,255] 

# 自动上色中最“暗”的部分。规则和base_colors相同。
# The 'darkest' color on bodies. All rules the same as base_colors. 
min_colors = [] 

# min_colors未填充部分的默认颜色。
# Default color for any unfilled or [] in min_colors. 
default_min_color = [0,0,0,255] 

# 自动生成贴图长宽（默认正方形贴图）
# 生成的贴图为texture_width*texture_amplifier x texture_width*texture_amplifier
# Texture width (and height). 
# The generated texture image is this*texture_amplifier x this*texture_amplifier
texture_width = 1024 

# 材质放大倍率。BB貌似支持这东西，比如您希望看到材质显示64x64但贴图有512x512那这玩意儿就是512/64=8.（且texture_width为64来生成512x512的贴图）
# UV image scaler supported in BB. e.g. if you want texture 64x64 but UV image 512x512 then this would be 512/64=8. (and texture_width would be 64 to generate 512x512 picture)
texture_amplifier = 1 # >=1 int. 

# 单个面所能占据的最大UV长度（最小为1）。所有面将根据该值缩放其在UV上的大小。
# 如果设置为0，将会尝试自动计算；如果设置为负数，将会使用最保守的/保证可以塞下所有面的大小（但可能会很小）
# Maximum length a single face can have on UV image. All faces will scale according to the longest surface. 
# If 0, will try to auto-detect; <0 will use the most conservative generation that garantees all faces can fit into the UV image (but could be very small). 
texture_max_width = 128 

# 每个UV色块边缘的空格宽度。2*这个+texture_max_width应当<=texture_width*texture_amplifier
#Empty pixels around each UV chunk. 2*this+texture_max_width should not exceed texture_width*texture_amplifier. 
uv_padding = 4 # >=0. 

# 光源角度向量[x,y,z]。会转算无需保证长度为1. MC模型一般使用 +y-z [0,1,-1]
# Location vector of light source when generating colors. 
# Unit length NOT required. Minecraft models usually use [0,1,-1]
light_source_angle = [0,1,-1] 

# 继承之前的贴图。v2再说。
# (WIP) Path to the old json file, if has one. The algorithm will try to inherite the UV mapping. 
# old_json = None 

# 注：动画并不保存在json里，只要组名（体名）相同BB里重新绑定一下动画就可以了。
# Note: Animations are NOT saved in json. You can re-link them in BB as long as you keep group (body) names the same. 

# 想看代码学习的话，必要前置知识点：三维向量运算，熟练使用numpy尤其是einsum
# For those would ilke to learn from my code, unescapable prereqs: 3D vector operations, numpy (especially einsum)
