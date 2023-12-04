# 导入所需模块
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import random
# plt显示彩色图片
def plt_showRGB(img):
#cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()
    
# plt显示灰度图片
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show()

def plt_write(img, name):
    # 先清除当前显示的图像
    plt.clf()
    plt.imshow(img,cmap='gray')
    plt.savefig("output\\"+name.split("\\")[-1])

def plt_writeRGB(img, name):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.clf()
    plt.imshow(img)
    plt.savefig("output\\"+name.split("\\")[-1])


# 图像去噪灰度处理
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def operate_one_img(img_name):

    # 读取待检测图片
    origin_image = cv2.imread(img_name)
    # 复制一张图片，在复制图上进行图像操作，保留原图
    image = origin_image.copy()
    # 图像去噪灰度处理
    gray_image = gray_guss(image)
    # x方向上的边缘检测（增强边缘信息）
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX

    # 图像阈值化操作——获得二值化图
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # 显示灰度图像
    # plt_show(image)
    # 形态学（从图像中提取对表达和描绘区域形状有意义的图像分量）——闭操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 1)
    # 显示灰度图像
    # plt_show(image)

    # 腐蚀（erode）和膨胀（dilate）
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    #x方向进行闭操作（抑制暗细节）
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    #y方向的开操作
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    # 中值滤波（去噪）
    image = cv2.medianBlur(image, 21)
    # 显示灰度图像
    # plt_show(image)
    # 获得轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    nums = []
    
    for j, item in enumerate(contours):
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        # 根据轮廓的形状特点，确定车牌的轮廓位置并截取图像
        
        if (weight > (height * 3)) and (weight < (height * 5)):
            
            cut_image = origin_image[y:y + height, x:x + weight]
            plates.append((x,y,weight,height,cut_image.copy()))
            plt_writeRGB(cut_image, img_name+str(j)+"_car_plate.jpg")
            
        elif(weight>height*1.5 and weight<height*2.5):
            cut_image = origin_image[y:y + height, x:x + weight]
            nums.append((x,y,weight,height, cut_image.copy()))
            plt_writeRGB(cut_image.copy(), img_name+str(j)+"_plate.jpg")
    for plate in plates:
        match_one(plate[4], img_name, plate[0], plate[1])
        
    for num in nums:
        match_one(num[4], img_name, num[0], num[1])



def match_one(image, name, ix, iy):
    

    #车牌字符分割
    # 图像去噪灰度处理
    gray_image = gray_guss(image)
    # 图像阈值化操作——获得二值化图   
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    # plt_show(image)
    # 保存灰度图像
    plt_write(image, name+"_threshold_gray"+str(ix)+".jpg")
    #膨胀操作，使“苏”字膨胀为一个近似的整体，为分割做准备
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.dilate(image, kernel)
    # 保存灰度图像
    plt_write(image, name+"_threshold_erode_gray"+str(ix)+".jpg")
    # plt_show(image)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    word_images = []
    #对所有轮廓逐一操作
    if(contours==[]):
        return
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(weight)
        word.append(height)
        words.append(word)
    # 排序，车牌号有顺序。words是一个嵌套列表
    words = sorted(words,key=lambda s:s[0],reverse=False)   
    #word中存放轮廓的起始点和宽高
    plt.clf()
    for word in words:
        # 筛选字符的轮廓
        
        testimg = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        
        plt.imshow(testimg,cmap='gray')
        # 添加一个图像来展示
        
        if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 3.5)) and (word[2] > 25):
            
            splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            
            word_images.append(splite_image)
            # print(i)
    # print(words)
    plt.clf()
    if(len(word_images)==0):
        return
    
    for i,j in enumerate(word_images):  
        plt.subplot(1,len(word_images),i+1)
        plt.imshow(word_images[i],cmap='gray')
        plt.axis('off')
    
    plt.savefig(name+"_plate_gray_dilate_split"+str(ix)+".jpg")
    # plt.show()
    word_images_ = word_images.copy()
    # 调用函数获得结果
    if(len(word_images_)==7):
        
        
        result = template_matching(word_images_)
        # print(result)
        # "".join(result)函数将列表转换为拼接好的字符串，方便结果显示
        cars.append((ix,iy,"".join(result)))
        # print( "".join(result))
    else:
        result = template_match_number(word_images_)
        if(len(result)!=0):
            numbers.append((ix,iy,"".join(result)))
            # print( "".join(result))

#模版匹配
# 准备模板(template[0-9]为数字模板；)
template = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
            '藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',
            '青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']

# 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
def read_directory(directory_name):
    referImg_list = []
    for filename in os.listdir(directory_name):
        referImg_list.append(directory_name + "/" + filename)
    return referImg_list

# 获得中文模板列表（只匹配车牌的第一个字符）
def get_chinese_words_list():
    chinese_words_list = []
    for i in range(34,64):
        #将模板存放在字典中
        c_word = read_directory('./refer1/'+ template[i])
        chinese_words_list.append(c_word)
    return chinese_words_list
chinese_words_list = get_chinese_words_list()


# 获得英文模板列表（只匹配车牌的第二个字符）
def get_eng_words_list():
    eng_words_list = []
    for i in range(10,34):
        e_word = read_directory('./refer1/'+ template[i])
        eng_words_list.append(e_word)
    return eng_words_list
eng_words_list = get_eng_words_list()


# 获得英文和数字模板列表（匹配车牌后面的字符）
def get_eng_num_words_list():
    eng_num_words_list = []
    for i in range(0,34):
        word = read_directory('./refer1/'+ template[i])
        eng_num_words_list.append(word)
    return eng_num_words_list
eng_num_words_list = get_eng_num_words_list()


# 读取一个模板地址与图片进行匹配，返回得分
def template_score(template,image):
    #将模板进行格式转换
    template_img=cv2.imdecode(np.fromfile(template,dtype=np.uint8),1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    #模板图像阈值化处理——获得黑白图
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
#     height, width = template_img.shape
#     image_ = image.copy()
#     image_ = cv2.resize(image_, (width, height))
    image_ = image.copy()
    #获得待检测图片的尺寸
    height, width = image_.shape
    # 将模板resize至与图像一样大小
    template_img = cv2.resize(template_img, (width, height))
    # 模板匹配，返回匹配得分
    result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
    return result[0][0]

def template_match_number(word_images):
    results = []
    for index,word_image in enumerate(word_images):
        best_score = []
        for eng_num_word_list in eng_num_words_list:
            score = []
            match_time = 0
            for eng_num_word in eng_num_word_list:
                result = template_score(eng_num_word,word_image)
                score.append(result)
                match_time+=1
                if(match_time==100):
                    break
            best_score.append(max(score))
        i = best_score.index(max(best_score))
        # print(template[i])
        r = template[i]
        results.append(r)
    return results

# 对分割得到的字符逐一匹配
def template_matching(word_images):
    results = []
    for index,word_image in enumerate(word_images):
        if index==0:
            best_score = []
            
            for chinese_words in chinese_words_list:
                score = []
                match_time = 0
                for chinese_word in chinese_words:
                    result = template_score(chinese_word,word_image)
                    score.append(result)
                    match_time+=1
                    if(match_time==100):
                        break
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[34+i])
            r = template[34+i]
            results.append(r)
            continue
        if index==1:
            best_score = []
            for eng_word_list in eng_words_list:
                score = []
                match_time = 0
                for eng_word in eng_word_list:
                    result = template_score(eng_word,word_image)
                    match_time+=1
                    score.append(result)
                    if(match_time==100):
                        break
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[10+i])
            r = template[10+i]
            results.append(r)
            continue
        else:
            best_score = []
            for eng_num_word_list in eng_num_words_list:
                score = []
                match_time = 0
                for eng_num_word in eng_num_word_list:
                    result = template_score(eng_num_word,word_image)
                    score.append(result)
                    match_time+=1
                    if(match_time==100):
                        break
                best_score.append(max(score))
            i = best_score.index(max(best_score))
            # print(template[i])
            r = template[i]
            results.append(r)
            continue
    return results


imgs = [os.path.join(".\\pics", each)  for each in os.listdir("./pics")]

cars = []
numbers = []

dismp = {}
nummp = {}

for img in imgs:
    operate_one_img(img)
    print(f"在文件{img}中")
    for number in numbers:

        distance = 10000000.0
        match_number = ""
        for car in cars:
            xycar = np.array([float(car[0]), float(car[1])])
            xynumber = np.array([float(number[0]), float(number[1])])

            # Calculate Euclidean distance
            n_dis = np.linalg.norm(xycar - xynumber)
            
            if(n_dis < distance):
                distance = n_dis
                match_number = number[2]
            # 如果这个编号没有被匹配，则匹配
        if(nummp.get(match_number)==None):
            nummp.update({match_number:car})
            dismp.update({match_number: distance})
        else:
            if(distance < dismp.get(match_number)):
                dismp.update({match_number:distance})
                nummp.update({match_number:car})

    for number in numbers:
        if(nummp.get(number[2])==None):
            print(f"编号{number[2]}上没有停放车辆")
        else:
            print(f"编号{number[2]}上停着{nummp.get(number[2])[2]}")
    cars.clear()
    numbers.clear()

    dismp.clear()
    nummp.clear()

    
    
    
    
