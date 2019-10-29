#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


# In[ ]:


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


# In[ ]:


def get_SIFT(template):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(template, None)
    return cv2.KeyPoint_convert(kp), desc


# In[ ]:


def switch_x_y(array_xy):
    array_yx = [0,0]
    array_yx[0] = array_xy[1]
    array_yx[1] = array_xy[0]
    return array_yx


# In[ ]:


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3


# In[ ]:


def EuD(rst,tar):
    return np.sqrt((rst[0,0]-tar[0,0])**2+(rst[1,0]-tar[1,0])**2)


# In[ ]:


template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
target_list = []
for i in range(4):
    target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
    target_list.append(target)
target_list[0].shape


# In[ ]:


img = cv2.imread('./Hyun_Soo_template.jpg')


# In[ ]:


def find_match(img1, img2, tolerate_rate = 0.8):
    sift_1 = get_SIFT(img1)
    sift_2 = get_SIFT(img2)
    neigh = NearestNeighbors(2, 0.4)
    neigh.fit(sift_2[1])
    x1_to_x2 = []
    for i in range(sift_1[0].shape[0]):
        tmp_fit_point = neigh.kneighbors([sift_1[1][i,:]], 2, return_distance=True)
        if(tmp_fit_point[0][0][0]/tmp_fit_point[0][0][1]<tolerate_rate):
            x1_to_x2.append([i,tmp_fit_point[1][0][0]])
    neigh.fit(sift_1[1])
    x2_to_x1 = []
    for i in range(sift_2[0].shape[0]):
        tmp_fit_point = neigh.kneighbors([sift_2[1][i,:]], 2, return_distance=True)
        if(tmp_fit_point[0][0][0]/tmp_fit_point[0][0][1]<tolerate_rate):
            x2_to_x1.append([i,tmp_fit_point[1][0][0]])
    x2_to_x1_inv = list(map(lambda x: switch_x_y(x),x2_to_x1))
    common_list = intersection(x1_to_x2,x2_to_x1_inv)
    x1 = []
    x2 = []
    for item in common_list:
        x1.append(sift_1[0][item[0],:])
        x2.append(sift_2[0][item[1],:])
    return np.mat(x1), np.mat(x2)


# In[ ]:


def align_image_using_feature(x_1, x_2, ransac_thr = 10, ransac_iter = 1000):
    np.random.seed(5561)
    max_include_num = 0
    A = np.zeros(shape=(3,3))
    for times in range(ransac_iter):
        random_select_list = list(np.random.choice(range(x_1.shape[0]), size=4, replace=False))
        mat_A = np.zeros(shape=(6,6))
        mat_B = np.zeros(shape=(6,1))
        for i in range(3):
            point = random_select_list[i]
            up = x_1[point][0,0]
            vp = x_1[point][0,1]
            uf = x_2[point][0,0]
            vf = x_2[point][0,1]
            A_list = [up,vp,1,0,0,0,0,0,0,up,vp,1]
            tmp_mat_A = np.mat(A_list).reshape((2,6))
            tmp_mat_B = np.mat([uf,vf]).reshape((2,1))
            mat_A[i*2] = tmp_mat_A[0]
            mat_A[i*2+1] = tmp_mat_A[1]
            mat_B[i*2] = tmp_mat_B[0]
            mat_B[i*2+1] = tmp_mat_B[1]
        x_sol = np.dot(np.dot(np.linalg.inv(np.dot(mat_A.T,mat_A)),mat_A.T),mat_B)
        #x_sol = np.dot(np.linalg.inv(mat_A),mat_B)
        tmp_list = list(x_sol)
        tmp_list.append(np.array([0]))
        tmp_list.append(np.array([0]))
        tmp_list.append(np.array([1]))
        H_mat = np.mat(tmp_list).reshape((3,3))
        include_counter = 0
        for j in range(x_1.shape[0]):
            input_point = np.ones(shape=(3,1))
            input_point[0,0] = x_1[j][0,0]
            input_point[1,0] = x_1[j][0,1]
            target_point = np.ones(shape=(3,1))
            target_point[0,0] = x_2[j][0,0]
            target_point[1,0] = x_2[j][0,1]
            rst = np.dot(H_mat,input_point)
            error = EuD(rst,target_point)
            if (error<ransac_thr):
                include_counter = include_counter + 1
        if (include_counter>max_include_num):
            max_include_num = include_counter
            A = H_mat
    return np.array(A)


# In[ ]:


match_t_0= find_match(template,target_list[0])


# In[ ]:


fig = plt.figure(figsize=(16,16))
visualize_find_match(template,target_list[0],match_t_0[0],match_t_0[1],img_h=500)


# In[ ]:


best_mat_find = align_image_using_feature(match_t_0[0], match_t_0[1], 10, 1000)


# In[ ]:


best_mat_find


# In[ ]:


def warp_image(img, A, output_size,interpn = "False"):
    inverse_map_mat = np.zeros(output_size)
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            input_pos = np.ones(shape=(3,1))
            input_pos[0,0] = j
            input_pos[1,0] = i
            #output_pos = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),input_pos)
            output_pos = np.dot(np.linalg.inv(A),input_pos)
            prev_x = int(np.floor(output_pos[1,0]))
            prev_y = int(np.floor(output_pos[0,0]))
            if (prev_x>=0 and prev_x<output_size[0] and prev_y>=0 and prev_y<output_size[1]):
                inverse_map_mat[prev_x][prev_y] = img[i,j]
    
    #exist_x = []
    #exist_y = []
    #exist_z = []
    #need_to_add_point = []
    #for i in range(output_size[0]):
    #    for j in range(output_size[1]):
    #        if (inverse_map_mat[i][j] == 0):
    #            need_to_add_point.append([i,j])
    #        else:
    #            exist_x.append(i)
    #            exist_y.append(j)
    #            exist_z.append([inverse_map_mat[i][j]])
    #if (len(need_to_add_point)!=0):
    #    need_to_add_value = interpolate.interpn((exist_x,exist_y), exist_z, need_to_add_point, method='linear')
    #    for k in len(need_to_add_point):
    #        inverse_map_mat[need_to_add_point[k][0],need_to_add_point[k][1]] = need_to_add_value[k]
    
    if (interpn == True):
        for i in range(inverse_map_mat.shape[0]):
            for j in range(inverse_map_mat.shape[1]):
                if(inverse_map_mat[i,j] == 0.0):
                    #current_point = np.ones(shape=(3,1))
                    #current_point[0,0] = j
                    #current_point[1,0] = i
                    current_point = np.array([j,i,1])
                    add_point = np.floor(np.dot(np.array(A),current_point)).astype(int)
                    p = add_point[1]
                    q = add_point[0]
                    #try:
                    inverse_map_mat[i,j] = img[p][q]
                    #except:
                    #    print(i,j,p,q)
                else:
                    continue
    img_warped = inverse_map_mat
    return img_warped


# In[ ]:


warp_output = warp_image(target_list[0], best_mat_find, template.shape)


# In[ ]:


fig = plt.figure(figsize=(8,8))
plt.imshow(warp_output, cmap='gray')


# In[ ]:


def get_differential_filter():
    filter_x = [[1,0,-1],[1,0,-1],[1,0,-1]]
    filter_y = [[1,1,1],[0,0,0],[-1,-1,-1]]
    return filter_x, filter_y

def filter_image(im, filter):
    im_f = np.zeros((np.size(im,0),np.size(im,1)));
    #suppose the filter is N by N
    padding_len = int(np.ceil((np.size(filter,0)-1)/2));
    im = np.pad(im,((padding_len,padding_len),(padding_len,padding_len)),'constant');
    center_k = np.floor(np.size(filter,0)/2);
    center_l = np.floor(np.size(filter,1)/2);
    x_range_hund = int(np.floor((np.size(im,0)-1)/100))
    fil_type = 'Unknown'
    if (filter == get_differential_filter()[0]):
        fil_type = 'filter_x'
    elif (filter == get_differential_filter()[1]):
        fil_type = 'filter_y'
    elif (filter == get_gaussian_filter()):
        fil_type = 'filter_Gaussian'
    else:
        fil_type == 'Unknown'
    for i in range(1,np.size(im,0)-1):
        #Here is a timer to ensure the program is still running
        if (i % x_range_hund == 0):
            k = i // x_range_hund
            fin_k = k//2
            rem_k = 50 - fin_k
            print("\r","|"*fin_k+"."*rem_k,"Finish filter image with %s %i%% " %(fil_type,k), end="")
        for j in range(1,np.size(im,1)-1):
            v = 0;
            break_flag = False;
            for k in range(0,np.size(filter,0)):
                if(break_flag == True):
                        break;
                for l in range(0,np.size(filter,1)):
                    i1 = int(i + k - center_k);
                    j1 = int(j + l - center_l);
                    if (i1 < 0 or i1 > np.size(im,0) or j1 < 0 or j1 > np.size(im,1)):
                        break_flag = True;
                        break;
                    else:
                        v = v + (im[i1][j1])*(filter[k][l]);
            im_f[i-1][j-1] = v;
    im_filtered = im_f
    print("\n")
    return im_filtered


# In[ ]:


def align_image(template, target, A, eps = 0.05, maxiter = 100):
    p = A.flatten()[:6]
    p[0] = p[0] - 1
    p[4] = p[4] - 1
    filter_x, filter_y = get_differential_filter()
    grad_x = filter_image(template, filter_x)
    grad_y = filter_image(template, filter_y)
    Hessain_H = np.zeros(shape=(6,6))
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            A_list = [j,i,1,0,0,0,0,0,0,j,i,1]
            tmp_mat_A = np.mat(A_list).reshape((2,6))
            SDI = np.dot(np.mat([grad_x[i,j],grad_y[i,j]]),tmp_mat_A)
            Hessain_H = Hessain_H + np.dot(SDI.T,SDI)
    itr_time = 0
    p_trans_mat = A
    error_list = []
    while 1 :
        warp_rst = warp_image(target, p_trans_mat, template.shape,interpn = "False")
        F = np.zeros((6, 1))
        err_rst = template - warp_rst
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                delW_delp = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
                tmp = np.dot(np.array([grad_x[i, j], grad_y[i, j]]), delW_delp).reshape((6, 1))
                F = F + np.dot(tmp, err_rst[i, j])
        del_p = np.dot(np.linalg.inv(Hessain_H), F)
        tmp = del_p.tolist()
        tmp[0][0] = tmp[0][0]+1
        tmp[4][0] = tmp[4][0]+1
        tmp.append([0])
        tmp.append([0])
        tmp.append([1])
        del_p_mat = np.mat(tmp).reshape((3,3))
        p_trans_mat = np.dot(p_trans_mat,np.linalg.inv(del_p_mat))
        itr_time = itr_time + 1
        error_list.append(np.linalg.norm(err_rst))
        print("Now the loop runs for ",itr_time,"times"," and norm.del_p is ",np.linalg.norm(del_p)," and norm.error",np.linalg.norm(err_rst))
        if (np.linalg.norm(del_p)<eps or itr_time >= maxiter):
            break
    A_refined = p_trans_mat
    return A_refined,np.array(error_list)


# In[ ]:


refined_mat = align_image(template, target_list[0], best_mat_find, eps = 0.05, maxiter = 150)


# In[ ]:


refined_mat[1]


# In[ ]:


fig = plt.figure(figsize=(12,12))
visualize_align_image(template, target_list[0], best_mat_find, refined_mat[0], errors=refined_mat[1])


# In[ ]:


def track_multi_frames(template, img_list, ransac_thr = 10, ransac_iter = 50):
    refined_mat_list = []
    for item in range(len(img_list)):
        x_1, x_2 = find_match(template, target_list[item])
        best_mat = align_image_using_feature(x_1, x_2, ransac_thr,ransac_iter)
        tmp_warped = warp_image(target_list[item], best_mat, template.shape)
        refined_mat,_ = align_image(template, target_list[item], best_mat,eps = 0.05, maxiter = 5)
        refined_mat_list.append(refined_mat)
        tmp_warped_2 = warp_image(target_list[item], refined_mat, template.shape).astype('uint8222222222
        template = tmp_warped_2
    A_list = refined_mat_list
    return A_list


# In[ ]:


track_list = track_multi_frames(template, target_list, ransac_thr = 10, ransac_iter = 50)


# In[ ]:


fig = plt.figure(figsize=(16,16))
visualize_track_multi_frames(template, target_list, track_list)


# In[ ]:


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)

