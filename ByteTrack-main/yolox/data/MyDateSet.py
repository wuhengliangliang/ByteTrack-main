
import numpy as np
import mindspore.dataset as ds

class MyDataset:
    """自定义数据集类"""
    def __init__(self):
        """自定义初始化操作"""
        self.data = np.random.sample((5, 2))  # 自定义数据
        self.label1 = np.random.sample((5, 1))  # 自定义标签1
        self.label2 = np.random.sample((5, 1))  # 自定义标签2

    def __getitem__(self, index):
        """自定义随机访问函数"""
        return self.data[index], self.label1[index], self.label2[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self.data)
    def tarin(self):
        dataset_column_names = ["image", "labels", "pre_fg_mask", "is_inbox_and_inCenter"]
        laod_train = ds.GeneratorDataset(yolo_dataset, column_names=dataset_column_names,
                                 num_parallel_workers=min(8, num_parallel_workers),
                                 python_multiprocessing=True,
                                 shard_id=rank, num_shards=device_num, shuffle=True)
        ds = ds.batch(batch_size, drop_remainder=True)



# 实例化数据集类
dataset_generator = MyDataset()
# 加载数据集
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label1", "label2"], shuffle=False)



# 迭代访问数据集
for data in dataset.create_dict_iterator():
    print("data:[{:7.5f},".format(data['data'].asnumpy()[0]),
          "{:7.5f}]  ".format(data['data'].asnumpy()[1]),
          "label1:[{:7.5f}]".format(data['label1'].asnumpy()[0]),
          "label2:[{:7.5f}]".format(data['label2'].asnumpy()[0]))

# 打印数据条数
print("data size:", dataset.get_dataset_size())
