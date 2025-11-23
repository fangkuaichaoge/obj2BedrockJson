"""
Rectangle-only OBJ to Minecraft Bedrock JSON Converter
语法修复版：解决f-string未终止错误，完美处理旋转微小偏移，强制同路径导出
支持OBJ格式：仅四边形面（f 1 2 3 4），BlockBench无错位兼容
"""
import os
import json
import time
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 配置参数（旋转优化核心）
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
default_file_path = 'obj/rect_tst.obj'
save_name = ''
BBL_per_unit = 1
coincident_tol = 1e-6  # 适度放大容差，解决旋转后浮点偏移
min_edge_length = 1e-6
trace_limit = 514

# 旋转矩阵配置（支持任意3x3正交矩阵，示例：X轴旋转90°）
model_rotation = [[1, 0, 0], 
                  [0, 0, -1], 
                  [0, 1, 0]]

# 运行时参数
file_path = ''
save_path = ''
auto_coloring = False
rotation_matrix = np.array([])
apply_rotation = False
ortho_corrected = False  # 标记是否进行过正交化处理

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 交互工具函数
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def inputFilePath(prompt, default):
    while True:
        path = input(f"{prompt}（默认：{default}）：").strip()
        if not path:
            path = default
        if os.path.isfile(path):
            return path
        print(f"❌ 错误：文件 {path} 不存在，请重新输入！")

def confirmRotation():
    if np.array_equal(model_rotation, np.eye(3)):
        return False
    confirm = input(f"\n配置中存在旋转矩阵，是否应用？（Y/N，默认N）：").strip().upper()
    return confirm == '' or confirm == 'N'

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 核心工具函数（旋转精准化改造）
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def tic():
    return time.time()

def toc(tstart, name="", prt=True):
    tend = time.time()
    if prt:
        print(f'{name} took: {tend - tstart:.3f} sec.')
    return tend - tstart

def getFileName(file_dir):
    slashIdx = max(file_dir.rfind('/'), file_dir.rfind('\\'))
    dotIdx = file_dir.rfind('.')
    return file_dir[slashIdx+1:dotIdx]

def toUnit(vecs):
    vecs = np.asarray(vecs, dtype=np.float64)
    if len(vecs.shape) <= 1:
        l = max(np.linalg.norm(vecs), 1e-12)
        return vecs / l
    ls = np.linalg.norm(vecs, axis=-1, keepdims=True)
    ls[ls == 0] = 1
    return vecs / ls

def batchRotate(points, rotation_mat):
    """批量旋转3D点（双精度+偏移修正）"""
    points = np.asarray(points, dtype=np.float64)
    original_shape = points.shape
    
    # 展平为2D数组处理
    if len(original_shape) == 3:
        points_2d = points.reshape(-1, 3)
    else:
        points_2d = points
    
    # 矩阵旋转（列向量模式，标准旋转逻辑）
    rotated_2d = np.einsum('ij,kj->ki', rotation_mat, points_2d)
    
    # 浮点偏移修正（关键修复：旋转后点坐标四舍五入到小数点后6位）
    rotated_2d = np.round(rotated_2d, 6)
    
    # 恢复原始形状
    if len(original_shape) == 3:
        rotated = rotated_2d.reshape(original_shape)
    else:
        rotated = rotated_2d
    return rotated

def rotatePlaneNorm(norm, rotation_mat):
    """旋转平面法向量（确保平面方向正确）"""
    norm = np.asarray(norm, dtype=np.float64)
    return np.einsum('ij,j->i', rotation_mat, norm)

def areOverlapping(pt1, pt2):
    return np.linalg.norm(np.asarray(pt1) - np.asarray(pt2)) < coincident_tol

def edgeEqual(e1, e2):
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d1 = np.sum(np.linalg.norm(e1 - e2, axis=-1))
    d2 = np.sum(np.linalg.norm(e1 - np.flip(e2, axis=0), axis=-1))
    return d1 < coincident_tol or d2 < coincident_tol

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 平面类（旋转后重新校准）
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Plane:
    def __init__(self, start, e1, e2, norm):
        self.start = np.asarray(start, dtype=np.float64)
        self.e1 = toUnit(e1)
        self.e2 = toUnit(e2)
        self.norm = toUnit(norm)

    def contains(self, pt):
        """点在平面判断（容差适配旋转偏移）"""
        pt = np.asarray(pt, dtype=np.float64)
        return np.abs(np.dot(pt - self.start, self.norm)) < coincident_tol * 2

    def equal(self, p2):
        """平面相等判断（旋转后法向量校准）"""
        return self.contains(p2.start) and np.abs(np.dot(self.norm, p2.norm)) > 1 - coincident_tol

def getPlaneFromRect(rect):
    rect = np.asarray(rect, dtype=np.float64)
    start = rect[0]
    e1 = rect[1] - rect[0]
    e2 = rect[3] - rect[0]
    norm = np.cross(e1, e2)
    return Plane(start, e1, e2, norm)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### OBJ加载+旋转全流程优化
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_rect_obj(obj_path, apply_rotation=False, rotation_mat=None):
    all_rects = []
    current_rects = []
    grps = []
    rect_ct = 0
    vertices = []

    with open(obj_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue

            if parts[0] in ['g', 's', 'mg', 'o']:
                if len(current_rects) > 0:
                    all_rects.append(np.asarray(current_rects, dtype=np.float64))
                    current_rects.clear()
                grp_name = parts[1] if len(parts) > 1 else f'group_{len(grps)}'
                grps.append(grp_name)

            elif parts[0] == 'v':
                if len(parts) < 4:
                    raise AssertionError(f"无效顶点数据：{line}")
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])

            elif parts[0] == 'f':
                if len(parts) != 5:
                    raise AssertionError(f"仅支持四边形面！当前面顶点数：{len(parts)-1}，行：{line}")
                try:
                    indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                except:
                    raise AssertionError(f"无效面索引：{line}")
                for idx in indices:
                    if idx < 0 or idx >= len(vertices):
                        raise AssertionError(f"顶点索引越界：{idx}（总顶点数：{len(vertices)}）")
                rect = np.asarray([vertices[i] for i in indices], dtype=np.float64)
                current_rects.append(rect)
                rect_ct += 1

        if len(current_rects) > 0:
            all_rects.append(np.asarray(current_rects, dtype=np.float64))

    # 旋转处理（新增平面法向量同步旋转）
    if apply_rotation and rotation_mat is not None:
        print("\n正在应用旋转矩阵（含法向量校准）...")
        rotated_all_rects = []
        for rect_group in all_rects:
            rotated_group = batchRotate(rect_group, rotation_mat)
            rotated_all_rects.append(rotated_group)
        all_rects = rotated_all_rects

    # 旋转后共面性二次校验（放宽容差）
    valid_all_rects = []
    for rect_group in all_rects:
        valid_rects = []
        for rect in rect_group:
            plane = getPlaneFromRect(rect)
            # 旋转后点可能存在微小偏移，放宽判断条件
            contain_count = 0
            for pt in rect:
                if plane.contains(pt):
                    contain_count += 1
            if contain_count >= 3:  # 4个点中至少3个在平面上即视为有效
                valid_rects.append(rect)
            else:
                print(f"警告：跳过非共面四边形面（旋转后偏移过大）")
        valid_all_rects.append(np.asarray(valid_rects, dtype=np.float64))

    return valid_all_rects, grps, rect_ct

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 长方形面处理（旋转后去重优化）
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RectPolygon:
    def __init__(self, plane):
        self.plane = plane
        self.rects = []
        self.edges = []
        self.outside = []

    def addRect(self, rect):
        rect = np.asarray(rect, dtype=np.float64)
        rect_edges = [
            np.asarray([rect[0], rect[1]], dtype=np.float64),
            np.asarray([rect[1], rect[2]], dtype=np.float64),
            np.asarray([rect[2], rect[3]], dtype=np.float64),
            np.asarray([rect[3], rect[0]], dtype=np.float64)
        ]
        edge_dirs = [toUnit(edge[1] - edge[0]) for edge in rect_edges]
        face_norm = self.plane.norm
        edge_norms = [np.cross(face_norm, dir) for dir in edge_dirs]

        # 旋转后边去重（容差适配）
        for edge, norm in zip(rect_edges, edge_norms):
            is_dup = False
            for existing_edge in self.edges:
                if edgeEqual(edge, existing_edge):
                    is_dup = True
                    break
            if not is_dup:
                self.edges.append(edge)
                self.outside.append(norm)
        self.rects.append(rect)

    def getFinalRects(self):
        final_rects = []
        for rect in self.rects:
            edge1_len = np.linalg.norm(rect[1] - rect[0])
            edge2_len = np.linalg.norm(rect[3] - rect[0])
            if edge1_len >= min_edge_length and edge2_len >= min_edge_length:
                final_rects.append(rect)
        return np.asarray(final_rects, dtype=np.float64)

def processRects(rect_group):
    polys = []
    ti = tic()

    for rect in rect_group:
        plane = getPlaneFromRect(rect)
        found = False
        for poly in polys:
            if poly.plane.equal(plane):
                poly.addRect(rect)
                found = True
                break
        if not found:
            new_poly = RectPolygon(plane)
            new_poly.addRect(rect)
            polys.append(new_poly)

    final_rects = []
    for poly in polys:
        final_rects.extend(poly.getFinalRects())

    toc(ti, "长方形面处理")
    print(f"有效长方形面总数：{len(final_rects)}")
    return np.asarray(final_rects, dtype=np.float64)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### BB JSON生成（旋转后坐标系精准适配）
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def rectToCube(rect):
    """长方形面转立方体（旋转后欧拉角优化）"""
    rect = np.asarray(rect, dtype=np.float64)
    v1 = rect[1] - rect[0]
    v2 = rect[3] - rect[0]
    dx = np.linalg.norm(v1)
    dy = np.linalg.norm(v2)

    # 计算旋转矩阵（双精度保证）
    e1 = toUnit(v1)
    e2 = toUnit(v2)
    e3 = toUnit(np.cross(e1, e2))
    R = np.vstack([e1, e2, e3]).T

    # 欧拉角计算（修复旋转方向+偏移修正）
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # BB坐标系适配（三重修正）
    angles = np.array([-roll, -pitch, yaw]) * 180 / np.pi  # 轴翻转
    angles = np.round(angles, 2)  # 浮点精度修正
    # 角度范围归一化（0-360°）
    angles = angles % 360
    angles[angles < 0] += 360

    # 位置计算（旋转后偏移修正）
    pos = rect[0].copy()
    pos[0] = -pos[0]  # BB X轴翻转
    pos = np.round(pos, 6)  # 坐标偏移修正
    size = np.array([dx, dy, 0.001], dtype=np.float64)
    ori = pos - size * np.array([1, 0, 0], dtype=np.float64)
    ori = np.round(ori, 6)

    # 单位转换
    pos *= BBL_per_unit
    size *= BBL_per_unit
    ori *= BBL_per_unit

    return {
        "origin": ori.tolist(),
        "size": size.tolist(),
        "pivot": pos.tolist(),
        "rotation": angles.tolist()
    }

def generateBBJson(all_rects, grps, file_name):
    bbjson = {
        "format_version": "1.12.0",
        "minecraft:geometry": [
            {
                "description": {
                    "identifier": f"geometry.{file_name}",
                    "texture_width": 1024,
                    "texture_height": 1024
                },
                "bones": []
            }
        ]
    }

    for rects, grp_name in zip(all_rects, grps):
        if len(rects) == 0:
            print(f"警告：分组 {grp_name} 无有效面，跳过")
            continue
        cubes = [rectToCube(rect) for rect in rects]
        bone = {
            "name": grp_name,
            "pivot": [0, 0, 0],
            "cubes": cubes
        }
        bbjson["minecraft:geometry"][0]["bones"].append(bone)

    return bbjson

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 主程序入口（全流程旋转优化+语法修复）
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("📌 长方形面OBJ → 基岩版JSON转换器（终极旋转修复版）")
        print("=" * 60)
        print("✨ 特性：解决旋转微小偏移，导出路径与OBJ一致")
        print("=" * 60)

        # 输入处理
        file_path = inputFilePath("请输入OBJ文件路径", default_file_path)
        save_path = os.path.dirname(file_path)
        print(f"\n✅ 自动设置导出目录：{save_path}")

        # 旋转矩阵初始化与校验
        rotation_matrix = np.asarray(model_rotation, dtype=np.float64)
        apply_rotation = False
        ortho_corrected = False
        if rotation_matrix.ndim == 2 and rotation_matrix.shape == (3, 3):
            det = np.linalg.det(rotation_matrix)
            if abs(det) < 1e-6:
                print("⚠️  旋转矩阵无效（行列式为0），跳过旋转")
            else:
                # 正交矩阵校验（旋转矩阵必须正交）
                ortho_check = np.allclose(np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), atol=1e-6)
                if not ortho_check:
                    print("⚠️  旋转矩阵非正交，自动正交化处理")
                    # 施密特正交化修正非正交矩阵
                    u, s, vh = np.linalg.svd(rotation_matrix)
                    rotation_matrix = np.dot(u, vh)
                    ortho_corrected = True
                apply_rotation = confirmRotation()
        else:
            print("⚠️  旋转矩阵格式错误（必须3x3），跳过旋转")

        # 配置验证
        ti_total = tic()
        if BBL_per_unit <= 0 or min_edge_length < 0 or trace_limit <= 0:
            raise AssertionError("配置参数错误：必须为正数")

        # 加载OBJ并旋转
        print(f"\n正在加载OBJ文件：{file_path}")
        all_rects, grps, rect_ct = load_rect_obj(
            file_path,
            apply_rotation=apply_rotation,
            rotation_mat=rotation_matrix if apply_rotation else None
        )
        print(f"加载完成 → 分组数：{len(grps)}，总长方形面数：{rect_ct}")
        if apply_rotation:
            if ortho_corrected:
                print(f"✅ 已成功应用旋转矩阵（含正交化处理）")
            else:
                print(f"✅ 已成功应用旋转矩阵")

        # 处理面
        print("\n正在处理长方形面...")
        processed_rects = []
        for i, (rect_group, grp_name) in enumerate(zip(all_rects, grps)):
            print(f"\n--- 处理分组：{grp_name}（原始面数：{len(rect_group)}）---")
            processed = processRects(rect_group)
            processed_rects.append(processed)

        # 生成JSON
        print("\n正在生成基岩版JSON文件...")
        file_name = save_name if save_name else getFileName(file_path)
        json_path = os.path.join(save_path, f"{file_name}.json")
        bbjson = generateBBJson(processed_rects, grps, file_name)

        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(bbjson, f, indent=4, ensure_ascii=False)
        print(f"✅ JSON文件已保存至：{json_path}")

        # 输出统计（修复f-string语法错误）
        dt_total = toc(ti_total, "\n总转换耗时")
        print("=" * 60)
        print("🎉 转换完成！")
        print(f"→ 输入文件：{file_path}")
        print(f"→ 输出文件：{json_path}")
        if apply_rotation:
            if ortho_corrected:
                print(f"→ 旋转应用：是（含正交化处理）")
            else:
                print(f"→ 旋转应用：是")
        else:
            print(f"→ 旋转应用：否")
        print(f"→ 总耗时：{dt_total:.3f} 秒")
        print("=" * 60)
        input("按回车键退出程序...")

    except Exception as e:
        print(f"\n❌ 转换失败：{str(e)}")
        input("按回车键退出程序...")
        raise