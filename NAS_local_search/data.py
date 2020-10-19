import numpy as np
import pandas as pd
from numpy import genfromtxt
from keras.utils import to_categorical
import sklearn.datasets as sd
from scipy.sparse import csr_matrix
import tensorflow as tf

def load_sensit():
    print('loading sensIT data...')
    x, y = sd.load_svmlight_file('datasets/sensit_combined_scale.train')
    x_test, y_test =  sd.load_svmlight_file('datasets/sensit_combined_scale.test')
    x = csr_matrix(x).toarray()
    x_test = csr_matrix(x_test).toarray()
    y = y - 1
    y_test = y_test - 1
    print(x.shape, y.shape, x_test.shape, y_test.shape)

    # choose samples
    idxs = np.random.permutation(x.shape[0])
    p = int(0.5 * x.shape[0])
    x1 = x[idxs[:p]].astype('float32')
    y1 = y[idxs[:p]].astype('int32')
    x2 = x[idxs[p:]].astype('float32')
    y2 = y[idxs[p:]].astype('int32')

    # datatype
    x_test = x_test.astype('float32')
    y_test = y_test.astype('int32')

    y1 = np.eye(3)[y1]
    y2 = np.eye(3)[y2]
    y_test = np.eye(3)[y_test]

    print("dataset shape:", x1.shape, y1.shape, x2.shape, y2.shape, x_test.shape, y_test.shape)
    print('load is done.\n')
    return x1, y1, x2, y2, x_test, y_test

def load_covertype():
    print('loading covertype data...')
    my_data = np.genfromtxt('./datasets/covertype_csv.csv', delimiter=',')[1:]
    assert not np.any(np.isnan(my_data))

    # train & test
    idxs = np.random.permutation(my_data.shape[0])
    p = int(0.9 * my_data.shape[0])
    x = my_data[idxs[:p], :-1]
    y = my_data[idxs[:p], -1] - 1
    x_test = my_data[idxs[p:], :-1]
    y_test = my_data[idxs[p:], -1] - 1

    # choose samples
    idxs = np.random.permutation(x.shape[0])
    p = int(0.5 * x.shape[0])
    x1 = x[idxs[:p]].astype('float32')
    y1 = y[idxs[:p]].astype('int32')
    x2 = x[idxs[p:]].astype('float32')
    y2 = y[idxs[p:]].astype('int32')

    # set datatype
    x_test = x_test.astype('float32')
    y_test = y_test.astype('int32')

    y1 = np.eye(7)[y1]
    y2 = np.eye(7)[y2]
    y_test = np.eye(7)[y_test]

    print("dataset shape:", x1.shape, y1.shape, x2.shape, y2.shape, x_test.shape, y_test.shape)
    print('load is done.\n')
    return x1, y1, x2, y2, x_test, y_test

def preprocess_data(data, useful_labels=None, flag=True):
    data.columns = [
        'duration',  # 持续时间，范围是 [0, 58329]
        'protocol_type',  # 协议类型，三种：TCP, UDP, ICMP
        'service',  # 目标主机的网络服务类型，共有70种，如‘http_443′,‘http_8001′,‘imap4′等
        'flag',  # 连接正常或错误的状态，离散类型，共11种，如‘S0′,‘S1′,‘S2′等
        'src_bytes',  # 从源主机到目标主机的数据的字节数，范围是 [0,1379963888]
        'dst_bytes',  # 从目标主机到源主机的数据的字节数，范围是 [0.1309937401]
        'land',  # 若连接来自/送达同一个主机/端口则为1，否则为0
        'wrong_fragment',  # 错误分段的数量，连续类型，范围是[0,3]
        'urgent',  # 加急包的个数，连续类型，范围是[0,14]
        'hot',  # 访问系统敏感文件和目录的次数，范围是[0,101]
        'num_failed_logins',  # 登录尝试失败的次数，范围是[0,5]
        'logged_in',  # 成功登录则为1，否则为0
        'num_compromised',  # compromised条件出现的次数，范围是[0,7479]
        'root_shell',  # 若获得root shell则为1，否则为0
        'su_attempted',  # 若出现”su root”命令则为1，否则为0
        'num_root',  # root用户访问次数，范围是[0,7468]
        'num_file_creations',  # 文件创建操作的次数，范围是[0,100]
        'num_shells',  # 使用shell命令的次数，范围是[0,5]
        'num_access_files',  # 访问控制文件的次数，范围是[0,9]
        'num_outbound_cmds',  # 一个FTP会话中出站连接的次数，数据集中这一特征出现次数为0。
        'is_host_login',  # 登录是否属于“hot”列表，是为1，否则为0
        'is_guest_login',  # 若是guest 登录则为1，否则为0
        'count',  # 过去两秒内，与当前连接具有相同的目标主机的连接数，范围是[0,511]
        'srv_count',  # 过去两秒内，与当前连接具有相同服务的连接数，范围是[0,511]
        'serror_rate',  # 过去两秒内，在与当前连接具有相同目标主机的连接中，出现“SYN” 错误的连接的百分比，范围是[0.00,1.00]
        'srv_serror_rate',  # 过去两秒内，在与当前连接具有相同服务的连接中，出现“SYN” 错误的连接的百分比，范围是[0.00,1.00]
        'rerror_rate',  # 过去两秒内，在与当前连接具有相同目标主机的连接中，出现“REJ” 错误的连接的百分比，范围是[0.00,1.00]
        'srv_rerror_rate',  # 过去两秒内，在与当前连接具有相同服务的连接中，出现“REJ” 错误的连接的百分比，范围是[0.00,1.00]
        'same_srv_rate',  # 过去两秒内，在与当前连接具有相同目标主机的连接中，与当前连接具有相同服务的连接的百分比，范围是[0.00,1.00]
        'diff_srv_rate',  # 过去两秒内，在与当前连接具有相同目标主机的连接中，与当前连接具有不同服务的连接的百分比，范围是[0.00,1.00]
        'srv_diff_host_rate',  # 过去两秒内，在与当前连接具有相同服务的连接中，与当前连接具有不同目标主机的连接的百分比，范围是[0.00,1.00]
        'dst_host_count',  # 前100个连接中，与当前连接具有相同目标主机的连接数，范围是[0,255]
        'dst_host_srv_count',  # 前100个连接中，与当前连接具有相同目标主机相同服务的连接数，范围是[0,255]
        'dst_host_same_srv_rate',  # 前100个连接中，与当前连接具有相同目标主机相同服务的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_diff_srv_rate',  # 前100个连接中，与当前连接具有相同目标主机不同服务的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_same_src_port_rate',  # 前100个连接中，与当前连接具有相同目标主机相同源端口的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_srv_diff_host_rate',  # 前100个连接中，与当前连接具有相同目标主机相同服务的连接中，与当前连接具有不同源主机的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_serror_rate',  # 前100个连接中，与当前连接具有相同目标主机的连接中，出现SYN错误的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_srv_serror_rate',  # 前100个连接中，与当前连接具有相同目标主机相同服务的连接中，出现SYN错误的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_rerror_rate',  # dst_host_rerror_rate. 前100个连接中，与当前连接具有相同目标主机的连接中，出现REJ错误的连接所占的百分比，范围是[0.00,1.00]
        'dst_host_srv_rerror_rate',  # 前100个连接中，与当前连接具有相同目标主机相同服务的连接中，出现REJ错误的连接所占的百分比，范围是[0.00,1.00]
        'outcome'  # 标签
    ]
    # print(data.sample(10))

    # 将数字值编码为z分数
    def encode_numeric_zscore(df, name, mean=None, sd=None):
        if mean is None:
            mean = df[name].mean()
        if sd is None:
            sd = df[name].std()
        df[name] = (df[name] - mean) / sd

    # 将文本值编码为虚拟变量(即[1,0,0],[0,1,0],[0,0,1]代表了红色、绿色、蓝色)
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name])
        for x in dummies.columns:
            dummy_name = f"{name}-{x}"
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)

    # 对每一项数据进行相应处理
    encode_numeric_zscore(data, 'duration')
    encode_text_dummy(data, 'protocol_type')
    encode_text_dummy(data, 'service')
    encode_text_dummy(data, 'flag')
    encode_numeric_zscore(data, 'src_bytes')
    encode_numeric_zscore(data, 'dst_bytes')
    encode_text_dummy(data, 'land')
    encode_numeric_zscore(data, 'wrong_fragment')
    encode_numeric_zscore(data, 'urgent')
    encode_numeric_zscore(data, 'hot')
    encode_numeric_zscore(data, 'num_failed_logins')
    encode_text_dummy(data, 'logged_in')
    encode_numeric_zscore(data, 'num_compromised')
    encode_numeric_zscore(data, 'root_shell')
    encode_numeric_zscore(data, 'su_attempted')
    encode_numeric_zscore(data, 'num_root')
    encode_numeric_zscore(data, 'num_file_creations')
    encode_numeric_zscore(data, 'num_shells')
    encode_numeric_zscore(data, 'num_access_files')
    encode_numeric_zscore(data, 'num_outbound_cmds')
    encode_text_dummy(data, 'is_host_login')
    encode_text_dummy(data, 'is_guest_login')
    encode_numeric_zscore(data, 'count')
    encode_numeric_zscore(data, 'srv_count')
    encode_numeric_zscore(data, 'serror_rate')
    encode_numeric_zscore(data, 'srv_serror_rate')
    encode_numeric_zscore(data, 'rerror_rate')
    encode_numeric_zscore(data, 'srv_rerror_rate')
    encode_numeric_zscore(data, 'same_srv_rate')
    encode_numeric_zscore(data, 'diff_srv_rate')
    encode_numeric_zscore(data, 'srv_diff_host_rate')
    encode_numeric_zscore(data, 'dst_host_count')
    encode_numeric_zscore(data, 'dst_host_srv_count')
    encode_numeric_zscore(data, 'dst_host_same_srv_rate')
    encode_numeric_zscore(data, 'dst_host_diff_srv_rate')
    encode_numeric_zscore(data, 'dst_host_same_src_port_rate')
    encode_numeric_zscore(data, 'dst_host_srv_diff_host_rate')
    encode_numeric_zscore(data, 'dst_host_serror_rate')
    encode_numeric_zscore(data, 'dst_host_srv_serror_rate')
    encode_numeric_zscore(data, 'dst_host_rerror_rate')
    encode_numeric_zscore(data, 'dst_host_srv_rerror_rate')

    # 因为每行都存在num_outbound_cmds=0，因此需要滤除缺失数据
    data.dropna(inplace=True, axis=1)
    # 各类别统计展示
    labels = data.outcome.value_counts()
    print(labels)
    if flag:
        useful_labels = labels.index.to_numpy()[:3]
    # if not flag:
    #     temp = np.array(useful_labels)
    #     useful_labels[0] = temp[0]
    #     useful_labels[1] = temp[2]
    #     useful_labels[2] = temp[1]
    print(useful_labels)

    rule = data.outcome.isin(useful_labels)
    new_data = data[rule]

    # 将前41项作为输入数据x，最后一列标签列作为y
    x_columns = new_data.columns.drop('outcome')
    x = new_data[x_columns].values
    dummies = pd.get_dummies(new_data['outcome'])  # 分类
    # outcomes = dummies.columns
    # num_classes = len(outcomes)
    y = dummies.values

    return x, y, useful_labels


def load_intrusion():
    print('loading intrusion data...')
    data_train = pd.read_csv("./datasets/kddcup_train_10_percent_corrected.csv")
    data_test = pd.read_csv("./datasets/kddcup_test_corrected.csv")
    x, y, useful_labels = preprocess_data(data_train)
    x_test, y_test, _ = preprocess_data(data_test, useful_labels=useful_labels, flag=False)

    y_a = np.argmax(y, axis=1)
    indices_0 = np.where(y_a == 0)[0]
    indices_1 = np.where(y_a == 1)[0]
    indices_2 = np.where(y_a == 2)[0]

    idxs = np.random.permutation(indices_0.shape[0])
    p = int(0.5 * indices_0.shape[0])
    x1_0 = x[idxs[:p]].astype('float32')
    y1_0 = y[idxs[:p]].astype('int8')
    x2_0 = x[idxs[p:]].astype('float32')
    y2_0 = y[idxs[p:]].astype('int8')

    idxs = np.random.permutation(indices_1.shape[0])
    p = int(0.5 * indices_1.shape[0])
    x1_1 = x[idxs[:p]].astype('float32')
    y1_1 = y[idxs[:p]].astype('int8')
    x2_1 = x[idxs[p:]].astype('float32')
    y2_1 = y[idxs[p:]].astype('int8')

    idxs = np.random.permutation(indices_2.shape[0])
    p = int(0.5 * indices_2.shape[0])
    x1_2 = x[idxs[:p]].astype('float32')
    y1_2 = y[idxs[:p]].astype('int8')
    x2_2 = x[idxs[p:]].astype('float32')
    y2_2 = y[idxs[p:]].astype('int8')

    x1 = np.concatenate((x1_0, x1_1, x1_2), axis=0)
    y1 = np.concatenate((y1_0, y1_1, y1_2), axis=0)
    x2 = np.concatenate((x2_0, x2_1, x2_2), axis=0)
    y2 = np.concatenate((y2_0, y2_1, y2_2), axis=0)

    # datatype
    x_test = x_test.astype('float32')
    y_test = y_test.astype('int8')

    print("dataset shape:", x1.shape, y1.shape, x2.shape, y2.shape, x_test.shape, y_test.shape)
    print('load is done.\n')
    return x1, y1, x2, y2, x_test, y_test

# x1, y1, x2, y2, x_test, y_test = load_intrusion()