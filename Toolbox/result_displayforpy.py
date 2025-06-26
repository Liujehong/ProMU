import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import minmax_scale

def load_data(dataset):
    if dataset == 'samson':
        data = scipy.io.loadmat('samson_cycunet_result.mat')
        col, p, L = 95, 3, 156
    elif dataset == 'jasper':
        data = scipy.io.loadmat('jasper_cycunet_result.mat')
        col, p, L = 100, 4, 198
    else:
        raise ValueError('Unknown dataset')
    return data, col, p, L

def reshape_data(data, col, p, L):
    A = np.reshape(data['A'].astype(float), (p, col * col))
    abu_est = np.reshape(data['abu_est'].astype(float), (p, col * col))
    Y = np.reshape(data['Y'].astype(float), (L, col * col))
    return A, abu_est, Y

# Calculate the Root Mean Square Error (RMSE) for abundance estimation
# aRMSE: absolute RMSE
# Parameters:
# A: True abundance matrix
# abu_est: Estimated abundance matrix
# p: Number of endmembers
# col: Number of spectral bands
# Return value:
# The root mean square error of the abundance estimation
#Hyperspectral Unmixing Using Deep Convolutional Autoencoders in a Supervised Scenario
def aRMSE(a, a_pred, p, col):
    """
    计算平均根均方误差（Root Mean Square Error，RMSE）。

    参数:
    A: 真实值矩阵，其中每一行代表一个样本的实际值。 (p, col * col)
    abu_est: 预测值矩阵，与A具有相同的行数和列数，每一行代表一个样本的预测值。
    p: 样本总数。
    col: 特征总数。

    返回值:
    平均根均方误差的值，表示预测值与真实值之间的平均差异程度。
    """
    if len(a.shape)!=2:
        a = a.reshape(p,col*col)
    if len(a_pred.shape)!=2:
        a_pred = a_pred.reshape(p,col*col)
    amrse = np.zeros(p)
    for i in range(p):
        amrse[i] = np.sqrt((np.sum((a[i,:] - a_pred[i,:])**2))  / ( col *col ) )
        print(f'{i}' +"th——"+ "armse: {}".format(amrse[i]))
    # return (np.sqrt((np.sum((a - a_pred)**2))  / ( col *col*p ) )) 
    return amrse.mean()
    # return np.sqrt(np.sum((a - a_pred)**2)  / ( col *col ) ) / p

def rRMSE(X, X_RE, L, col):
    """
Calculate the Spectral Angle Mapper (SAM) for evaluation
SAM: spectral angle mapper
Parameters:
X: Original hyperspectral data matrix dim:(p,N) p~num_endmembers, N~num_pixels
X_RE: Reconstructed hyperspectral data matrix dim:(L,N) L~num_bands, N~num_pixels
L: Number of pixels
col: Number of spectral bands
Return value:
The average spectral angle mapper value
#average spectral angle mapper
"""
    if len(X.shape)!=2:
        X = X.squeeze(3).squeeze(2).transpose(1,0).reshape(L,col*col)
    if len(X_RE.shape)!=2:
        X_RE = X_RE.squeeze(3).squeeze(2).transpose(1,0).reshape(L,col*col)
    return np.sum(np.sqrt(np.sum((X - X_RE ) ** 2, 1) /( col *col) )) / L 



def SAM(X, X_RE, L, col):
    """
    计算相似性角度指标（Similarity Angle Measure，SAM）。

    SAM用于衡量两个向量集合之间的相似性，通过计算两个集合的夹角余弦的反余弦值之和来实现。
    
    参数:
    X: 原始向量集合，二维数组。(L, col * col)
    X_RE: 变换后的向量集合，二维数组。
    L: 用于规范化向量长度的因子，一维数组。
    col: 向量集合中的向量数量，整数。

    返回值:
    SAM值，一个浮点数，表示两个向量集合之间的相似性程度。
    """
    # 计算X和X_RE的内积，并提取对角线元素
    # 这里使用了numpy的@操作符进行矩阵乘法，然后使用diagonal函数提取对角线元素
    # 接着，通过除以X和X_RE的各自范数之积来获得夹角余弦值
    # 最后，对所有夹角余弦值的反余弦值求和，并除以col的平方，得到SAM值
    if len(X.shape)!=2:
        X = X.squeeze(3).squeeze(2).transpose(1,0).reshape(L,col*col)
    if len(X_RE.shape)!=2:
        X_RE = X_RE.squeeze(3).squeeze(2).transpose(1,0).reshape(L,col*col)
    SAM = np.sum(np.arccos(np.diagonal(X.T @ X_RE) / ((np.linalg.norm(X,ord=2, axis = 0)) *  (np.linalg.norm(X_RE,ord=2, axis = 0))))) /(col*col)
    return SAM

def eSAM(e, e_pred, L, p):
    """
    计算相似性角度指标（Similarity Angle Measure，SAM）。

    SAM用于衡量两个向量集合之间的相似性，通过计算两个集合的夹角余弦的反余弦值之和来实现。
    
    参数:
    X: 原始向量集合，二维数组。(L, p) L band of number ,p denote the number of endmembers
    X_RE: 变换后的向量集合，二维数组。
    L: 用于规范化向量长度的因子，一维数组。
    col: 向量集合中的向量数量，整数。

    返回值:
    SAM值，一个浮点数，表示两个向量集合之间的相似性程度。
    """
    e = minmax_scale(e)
    e_pred = minmax_scale(e_pred)
    # 计算SAM
    SAM = np.zeros(p)
    for i  in range(p):
        s = np.sum(np.dot(e[:,i],e_pred[:,i]))
        t = np.sqrt(np.sum(e[:,i]**2))*np.sqrt(np.sum(e_pred[:,i]**2))
        SAM[i] = np.arccos(s/(t))
        print(f'{i}' + "th——"+"SAM: {}".format(SAM[i]))
    # print(SAM)
    avgSAM = np.mean(SAM)
    return avgSAM




# # 选择数据集
# dataset = 'samson'
# display_if = True

# # 加载数据
# data, col, p, L = load_data(dataset)

# # 数据重整
# A, abu_est, Y = reshape_data(data, col, p, L)

# # 端元估计
# M_est = EndmemberEst(Y, abu_est, 300)

# # # 评估指标
# # aRMSE_value = aRMSE(A, abu_est, p, col)
# # rRMSE_value = rRMSE(Y, X, p, col)
# # SAM,aSAM = aSAM(Y, X, p, col)
# # ESAM, aESAM = aESAM(M_est, data['M'], p, col)
# # if dataset == 'samson':
# #     print(f'aRMSE: {aRMSE_value}')
# #     print(f'ESAM_Water: {ESAM[0]}, ESAM_Tree: {ESAM[1]} , ESAM_Soil: {ESAM[2]}')
# #     print(f'aESAM: {aESAM}')

# # jasper 待修改！
# # elif dataset == 'jasper':
# #     print(f'RMSE: {rmse_value}')
# #     print(f'SAD_Water: {SAD[0]}, SAD_Tree: {SAD[1]} , SAD_Soil: {SAD[2]}, SAD_Grass: {SAD[3]}')
# #     print(f'SADerr: {SADerr}')
# # # 估计丰度图像
# if display_if:
#     plt.figure()
#     abu_cube = np.reshape(abu_est.T, (col, col, p))
#     for i in range(p):
#         plt.subplot(2, p, i + 1)
#         plt.imshow(abu_cube[:, :, i].T, aspect='auto',cmap='jet')
#         plt.axis('off')
#     # plt.set_cmap('jet')

# # 显示实际丰度图像
# if display_if:
#     gt = np.reshape(A.T, (col, col, p))
#     for i in range(p):
#         plt.subplot(2, p, i + p + 1)
#         plt.imshow(gt[:, :, i].T, aspect='auto',cmap='jet')
#         plt.axis('off')
#     # plt.set_cmap('jet')
# plt.savefig(f'{dataset}_result.png')
# plt.close()

# if display_if:
#     # wavelengths = M_pred.shape[1]  # 波长
#     plt.figure()
#     # 使用 plot 函数绘制光谱
#     plt.plot(M_est[:,i],color='red')
#     plt.plot(data['M'][:, i],color='blue')
#     # cbar_pred = plt.colorbar(im_pred)
#     # 显示图形
#     plt.savefig('output_M_reslut.png')

    
