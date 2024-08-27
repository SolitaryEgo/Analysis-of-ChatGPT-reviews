import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import nltk
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords


data = pd.read_csv('./chatgpt_reviews.csv')
print(data.head())

print(data.isnull().sum())

# 填充缺失值
data['reviewCreatedVersion'] = data['reviewCreatedVersion'].fillna('Unknown')
data['appVersion'] = data['appVersion'].fillna('Unknown')
data['userName'] = data['userName'].fillna('Unknown')
data['content'] = data['content'].fillna('No review provided')

print(data.isnull().sum())

df = data[['content']].copy()

nltk.data.path.append('E:/nltk_data')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()  # 删除多余空格和转行符
    text = re.sub(r'[^\w\s]', ' ', text)  # 删除标点符号
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # 删除停用词
    return ' '.join(tokens)  # 将分词重新组合单个字符串


df.loc[:, 'content'] = df['content'].apply(preprocess_text)

# 设置停用词
stopwords = set(STOPWORDS)
stopwords.update(['chatgpt', 'app', 'apps', 'gpt', 'review', 'reviews', 'ChatGPT'])

# 计算文本长度
print(df['content'].apply(len).describe())

print(f'评论数量：{df.shape[0]}')

df.loc[:, 'review_length'] = df['content'].apply(len)
print(f'平均评论长度：{df["review_length"].mean()}')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

plt.figure(figsize=(10, 6))
sns.histplot(df['review_length'], bins=30, kde=True)  # 使用 kde=True 来启用核密度估计
plt.title('评论长度分布')
plt.xlabel('评论长度')
plt.ylabel('频率')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['score'], bins=5, kde=True)  # 同样，使用 kde=True
plt.title('评论得分分布')
plt.xlabel('得分')
plt.ylabel('频率')
plt.show()

text = ' '.join(df['content'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=stopwords
).generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
