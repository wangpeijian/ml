import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset/iris.data", names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类型'])

x = df.loc[0:, ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']]
y = df.loc[0:, ['类型']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=40)

clf = RandomForestClassifier(max_depth=5, n_estimators=30)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

result = pd.DataFrame(data=x_test, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])

result.insert(loc=4, column='实际类别', value=y_test['类型'])
result.insert(loc=5, column='预测类别', value=y_predict)
result['正确'] = result['实际类别'] == result['预测类别']
result['succ'] = result['正确'].map({True: 1, False: 0})

print(result)
print("准确率:", result['succ'].sum() / result['succ'].count())
