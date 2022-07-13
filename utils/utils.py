from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

def Evaluation(true, pre):
    """
    计算评价指标的函数
    评价指标：
    Accuracy（准确率）：预测对的样本总数的比例
    Precision（精确率）：针对预测结果而言，预测为正的样本中是有多少是真正的正样本。需要加上,average='micro'
    Recall（召回率）：针对我们原来的样本而言，样本中的正例有多少被预测正确了
    """
    
    Accuracy = accuracy_score(true, pre)
    Precision = precision_score(true, pre, average='micro')
    Recall = recall_score(true, pre, average='micro')
    F1Score = f1_score(true, pre, average='micro')

    return round(Accuracy, 4), round(Precision, 4), round(Recall, 4), round(F1Score, 4)

def plot_eval(epoch, train_loss,test_loss, train_acc, test_acc, output_path):

    # 创建x轴，epoch步数
    x = []
    for i in range(epoch + 1):
        x.append(i)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, train_acc, color="g",label="Train Acc")
    ax1.plot(x, test_acc, color="r", label="Test Acc")
    ax2.plot(x, train_loss, color="g", label="Train Loss")
    ax2.plot(x, test_loss, color="r", label="Test Loss")
    ax1.legend()  # 添加图例
    ax2.legend()  # 添加图例

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Acc",color='g')
    ax2.set_ylabel("Loss",color='b')
    
    output_path = os.path.join(output_path, "eval.jpg")
    plt.savefig(output_path)


if __name__ == "__main__":
    # 测试用
    Evaluation([1,2,3], [2,3,3])