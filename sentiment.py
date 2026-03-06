# sentiment.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ---------- 1. 加载数据 ----------
df = pd.read_csv('中文酒店评论数据集.csv')

# 检查并处理缺失值
print("原始数据形状:", df.shape)
print("缺失值统计:\n", df.isnull().sum())
df = df.dropna(subset=['review'])  # 删除 review 为空的记录
print("处理后数据形状:", df.shape)

X = df['review']
y = df['label']

print("数据集总样本数:", len(df))
print("标签分布:\n", df['label'].value_counts())
# ---------- 2. 分割训练集和测试集 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("训练集标签分布:\n", y_train.value_counts())
print("测试集标签分布:\n", y_test.value_counts())
# ---------- 3. 构建模型流水线 ----------
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),  # 把文本转成数字特征
    ('clf', LinearSVC())                            # 分类器
])

# ---------- 4. 训练模型 ----------
print("正在训练模型...")
model.fit(X_train, y_train)

# ---------- 5. 评估准确率 ----------
accuracy = model.score(X_test, y_test)
print(f"模型在测试集上的准确率: {accuracy:.3f}")
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# ---------- 6. 自己测试几个句子 ----------
test_sentences = [
    "这家店太棒了，下次还来",
    "环境很差，服务态度也不好",
    "价格实惠，味道不错",
    "不会再来了，失望"
]
predictions = model.predict(test_sentences)
print("\n预测结果：")
for sentence, pred in zip(test_sentences, predictions):
    print(f"“{sentence}” → {pred}")