import test2
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_excel('第8组数据.xlsx')
X = df.drop(columns=(['输出参数1', '序号', '输出参数2']))
y = df['输出参数1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
population_size = 20
chromosome_num = 2  # 2
max_value = 10
chromosome_length = 20
iter_num = 100
pc = 0.6
pm = 0.01
res=test2.diaoyong(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm,
             X_train, X_test, y_train, y_test)
print(res)