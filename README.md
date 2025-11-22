文件列表 Files should be included:
obj
|-- tst.obj
json
config.py
Main.py
mathHelper.py
objProcessor.py
plane.py
polygon.py
rectFc.py
shader.py
toBedrock.py
trigOp.py
readme.md
requirement.txt

项目链接 Project link
https://github.com/Yinhe2332767/obj2BedrockJson

作者b站 Author on bilibili:
https://space.bilibili.com/38833357?spm_id_from=333.1007.0.0

食用方法：
-1. 安装Python（可能还有VScode）
0. 安装requirement.txt内的依赖库，可直接运行Main.py检验安装
1. 在config.py中填写文件路径并调整参数（默认一般能用。有需要自己调。详见文件内说明）
2. 运行Main.py
3. 面数高的模型要转一会儿，可以挂后台去忙自己的事，过段时间回来收文件

How to use:
-1. Install Python. (Also VScode maybe)
0. Install all dependencies in requirement.txt. You may run Main.py OOB to check installation.
1. Fill in file path and config parameters in config.py (Default usually works. Tone as needed. Further info inside the file)
2. Run Main.py
3. It takes a minute to convert complicated models. You can then minimize this window, go back to your business and come back later to see the result.

注意事项：
1. 该算法不能完美转换任意模型。这是受制于底层数学的（怎么用长方形拼成三角形嘛？），欧几里得空间内不可能突破。能被完美转换的模型必须本来就可以在BB里做出。
2. 暂不支持曲面估算。其实保存到obj时曲面已经被估算为三角面了，即到算法这一步是不可能知道哪些面原来是曲面的，因此估算曲面会极其困难。
3. 因为算法已经复杂成这个鬼样子了，目前算法不会处理同平面矩形不完全相交的情况，所以看到部分重叠是正常的，请勿回报错误。
4. 基于底层数学限制，该转换器将无法转出：（当同平面三角面合并后仍存在的）
|-- 三角面
|-- 多边形面中含有锐角的部分
5. 目前算法暂时不支持转换：
|-- 不可从外部直线看到的可见面（仅考虑自身所在体）
6. 目前算法暂时不一定能转换：
|-- 所有边外夹角<90°的“齿轮”面（可尝试增加vis\_sample\_points）
|-- 自身所在体范围内只能从有限角度看到的可见面（可尝试增加vis\_sample\_points）
7. 因为是作者一人单挑的复杂程序，目前难免会出现问题。如果碰到确认属于算法本身而不是模型的问题，请在bilibili或github上回报作者。
|-- 已观测到的一个神奇问题，如果您在WSL内运行转换，BB可能无法导入算法生成的UV贴图。将贴图复制回Windows系统下即可正常运行。

Notes:
1. It is NOT garanteed that this algorithm can convert all models perfectly. It is a mathematically limit and thus unbreakable that you can not perfectly represent a triangular surface with any amount of rectangles, within Euclidian space. To perfectly convert any model, it must be buildable in Blockbench.
2. Curved surfaces are NOT supported. When saved to obj, all curveds are already estimated to triangles. Thus, it is not possible to know which surfaces are curved within this algorithm. The estimation will therefore be extremely complicated.
3. It is expected to see partially overlapping rectangles. Please do not report that as an error. The complexity of adding that function is beyond the scope.
4. Due to the mathematical limit, the converter can NEVER handle: (after all co-planer surfaces are merged)
|-- Triangular surfaces
|-- Parts containing angles <90deg in a polygon surface
5. Currently unsupported conversions:
|-- Surfaces not possible to see directly from outside (within its own body)
6. Currently not-that-supported conversions:
|-- 'Gear-shaped' surfaces that has all external angles <90deg (Can try to increase vis\_sample\_points if needed)
|-- Surfaces only visible from limited angles within their own bodies (Can try to increase vis\_sample\_points)
7. This was a one-person work, so errors are quite expected. If you met problems that you confirm is due to this algorithm along, please report to the author on Github or Bilibili.
|-- An observed problem you might encounter again: If you are using WSL, you may see that BB is not able to import texture image stored in WSL. Coping that image over to windows system would solve this issue.

用户协议：您可以使用本作来进行任何创作，包括商用目的，但您必须标明转换算法原作者为Yinhe233, uid38833357@bilibili; Yinhe2332767@github（两标至少取其一，b站标必须包括uid）
User agreement: You are allowed to use this code for anything, including business purposes, but you MUST label clearly that the converter algorithm is made by Yinhe233, uid38833357@bilibili; Yinhe2332767@github (at least one of two labels. The bilibili one must include my uid)
若未标明作者或标注不清*的盗用，作者保留依法追责权力。
I reserve my rights to hold those pirating the code, with no author label or unclear author labels*, legally responsible. 

* 标注清晰定义：必须将作者信息标注于简介或文字内容内单独一行，字号大于等于简介或文字内容内其他所有字的平均值，与任何情况背景像素RGB差值最小值不低于100（RGB按255最高值标注），如排版支持Alpha透明度则透明度不低于200（Alpha按255）；如果发布客体不支持简介或简介位于从主页起三级菜单及以上的（即需要点击三次及以上才能看到该信息），需标注于视频或图片封面。字体（不含背景，字与字的间距不得超过字横长的0.8倍，原先有空格处则为2倍）在首页状态下的占比不低于总像素量的1/500（大致举例：1080x960图片中加粗字体的方格至少占108x96或等面积的排版），同样必须有与背景（字体边缘向外10像素或图像最长轴的1/100中最大值，取最小差值）最小100的RGB差值及至少200 Alpha。对于简介、图片、视频等任何表达媒介都放不了的，如TACZ模型上，您可以不标注。（但是发布视频/专栏等需要标注）
* Detailed standard for clearly labeled: You MUST label the author information in a stand-along row in your description or text content, with no smaller font than the average of all other letters within this description or text content, and no smaller than 100 minimum value RGB color difference to the background, in RGB 255, in any case. If alpha (transparency) is supported, then the author label letters must have an alpha >= 200, in alpha 255. For where a description is not allowed or the description is hidden in or beyond a third level menu, i.e. you need to click at least three times to reach that description from the home page, the author label needs to be on the video or image cover, with the non-background part of the label occupaing at least 1/500 pixels area-wise, observed when the cover is on the home page. An rough example would be a label block with thicken font should occupy 108x96 or larger on an image 1080x960. Any letter may not be apart from another letter over 0.8 letter length, exception to where there is a space in the declaration above where 2 letter length is permitted. Also minimum 100 RGB difference to BG (Sampling the greater of 10 or longest axle of the image/100 number of pixels around the letters, counting the minimum value) and at least 200 alpha. For where none of description, image, video or any kind of espression is allowed, e.g. on a TACZ model, you may skip labelling. (but do please label algorithm author in you release video/blog/anything)

* 安啦写这么多只是用来防止*某些人*。只要你不是蓄意盗用，我不会找你麻烦的 :)
* Please don't be scared of the terms. It's just in case of *some people*. You'll be fine as long as you aren't pirating my code :)

* 注：这个用于blcokbench obj转json的所以我把所有的ture全改成了false
* 注：blockbench 的 obj 必须用blender 处理
* 优化了导入路径
