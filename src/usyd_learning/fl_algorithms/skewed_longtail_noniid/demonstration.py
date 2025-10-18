# 使用字典来模拟图中描述的数据分布
# 外层Key: 客户端ID (Client ID)
# 内层Key: 数据标签 (Label)
# 内层Value: 相对数据量 (Data Volume)

client_data_distribution = {
    # 客户端 0: 数据量小，只有标签 0 和 1
    0: {0: 10, 1: 10},
    
    # 客户端 1: 数据量小，只有标签 0 和 1
    1: {0: 10, 1: 15},
    
    # 客户端 2: 数据量较小，只有标签 2 和 3
    2: {2: 20, 3: 15},
    
    # 客户端 3: 数据量较小，只有标签 2 和 3
    3: {2: 25, 3: 25},

    # 客户端 4: 数据量中等，标签范围扩大
    4: {2: 20, 3: 20, 4: 15, 5: 10},

    # 客户端 5: 数据量中等，标签范围与4相似
    5: {2: 25, 3: 25, 4: 20, 5: 15},

    # 客户端 6: 数据量较大，标签范围继续扩大
    6: {2: 25, 3: 25, 4: 25, 5: 25, 6: 30},

    # 客户端 7: 数据量大，标签范围更广
    7: {2: 30, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30},

    # 客户端 8: 数据量非常大，覆盖几乎所有标签
    8: {0: 15, 1: 15, 2: 30, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 60},

    # 客户端 9: 数据量最大（长尾），覆盖所有标签
    9: {0: 20, 1: 20, 2: 30, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 60, 9: 100},
}

# 打印数据分布以验证
def print_distribution(data):
    print("--- Client Data Distribution ---")
    for client_id, labels in data.items():
        total_volume = sum(labels.values())
        print(f"Client ID: {client_id}")
        print(f"  - Labels Present: {list(labels.keys())}")
        print(f"  - Total Data Volume: {total_volume}")
        # print(f"  - Label Details: {labels}") # 取消注释可查看详细数据
    print("--------------------------------")

print_distribution(client_data_distribution)

# 用法示例（可选）：将上述稀疏字典作为 skewed long-tail 分布输入
# 实际使用时应提供一个真实的 DataLoader，然后调用分区器。
# 下面示例仅展示如何接入，不会在此文件中真实运行数据下载。
#
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from .skewed_longtail_partitioner import SkewedLongtailPartitioner, SkewedLongtailArgs
#
# if __name__ == "__main__":
#     # 1) 构建基础数据加载器（以 MNIST 为例）
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_ds = datasets.MNIST(root="../../../.dataset", train=True, download=True, transform=transform)
#     base_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)
#
#     # 2) 使用 skewed long-tail 分区器
#     partitioner = SkewedLongtailPartitioner(base_loader)
#     args = SkewedLongtailArgs(batch_size=64, shuffle=True, num_workers=0, return_loaders=True)
#     client_loaders = partitioner.partition(client_data_distribution, args)
#
#     # 3) 检查每个客户端的样本数
#     for i, ld in enumerate(client_loaders):
#         total = sum(len(b[0]) for b in ld)
#         print(f"Client {i}: total samples = {total}")
