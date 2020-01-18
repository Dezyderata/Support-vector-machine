from mpl_toolkits import mplot3d
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn import linear_model

input_file = 'ankieta.csv'
data = np.loadtxt(input_file, dtype='U25, U25, U32, U32, i4', delimiter=';')
np.random.shuffle(data)

sex_labels = ['Kobieta', 'Mężczyzna']
simple_labels = ['Nie', 'Tak']
multiple_labels = ['Nigdy', 'Rzadziej, niż kilka razy w roku', 'Kilka razy w roku', 'Około raz w miesiącu', 'Kilka razy w miesiącu', 'Kilka razy w tygodniu']

#papierosy, alkohol, slodycze, radosc 

sex, cigarettes, alcohol, sweets, happiness = zip(*data)

sex_encoder = preprocessing.LabelEncoder();
simple_encoder = preprocessing.LabelEncoder();
multiple_encoder = preprocessing.LabelEncoder();

sex_encoder.fit(sex_labels)
simple_encoder.fit(simple_labels)
multiple_encoder.fit(multiple_labels)

encoded_sex = sex_encoder.transform(sex)
encoded_cigarettes = simple_encoder.transform(cigarettes)
encoded_alcohol = multiple_encoder.transform(alcohol)
encoded_sweets = multiple_encoder.transform(sweets)

print('\nLable mapping sex:')
for i, item in enumerate(sex_encoder.classes_):
    print(item, '-->', i)

print('\nLable mapping simple:')
for i, item in enumerate(simple_encoder.classes_):
    print(item, '-->', i)

print('\nLable mapping multiple:')
for i, item in enumerate(multiple_encoder.classes_):
    print(item, '-->', i)

X = np.column_stack([encoded_sex, encoded_cigarettes,encoded_alcohol,encoded_sweets])
#print(list(X))

num_training = int(0.9*len(happiness))

X_tran, y_tran = X[:num_training], happiness[:num_training]
X_test, y_test = X[num_training:], happiness[num_training:]

regressor = linear_model.HuberRegressor(max_iter=2000)
regressor.fit(X_tran, y_tran)
y_predict = regressor.predict(X_test)

prediction_error = sm.mean_absolute_error(y_test, y_predict)
print("\nMean absolute error: ", prediction_error)

sex, cigarettes, alcohol, sweets = zip(*X_test)
decoded_sex = sex_encoder.inverse_transform(sex)
decoded_cigarettes = simple_encoder.inverse_transform(cigarettes)
decoded_alcohol = multiple_encoder.inverse_transform(alcohol)
decoded_sweets = multiple_encoder.inverse_transform(sweets)

for i in range(len(y_test)):
    print(f'\n{i} \nSex: {sex[i]} --> {decoded_sex[i]}')
    print(f'Alkohol: {alcohol[i]} --> {decoded_alcohol[i]}')
    print(f'Papierosy: {cigarettes[i]} --> {decoded_cigarettes[i]}')
    print(f'Slodycze: {sweets[i]} --> {decoded_sweets[i]}')
    print(f'Zadowolenie przewidziane: {y_predict[i]}, zadowolenie zmierzone: {y_tran[i]}')
fig = plt.figure()
ax = plt.axes(projection='3d')
zline = encoded_alcohol[num_training:]
xline = encoded_sweets[num_training:]
ax.set_xlabel('sweets')
ax.set_zlabel('happiness')
ax.set_ylabel('alcohol')
ax.scatter3D(xline, zline, y_test, color='red')
ax.scatter3D(xline, zline, y_predict, color='black')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
zline = encoded_alcohol[num_training:]
xline = encoded_cigarettes[num_training:]
ax.set_xlabel('cigarettes')
ax.set_zlabel('happiness')
ax.set_ylabel('alcohol')
ax.scatter3D(xline, zline, y_test, color='red')
ax.scatter3D(xline, zline, y_predict, color='black')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
zline = encoded_sweets[num_training:]
xline = encoded_cigarettes[num_training:]
ax.set_xlabel('cigarettes')
ax.set_zlabel('happiness')
ax.set_ylabel('sweets')
ax.scatter3D(xline, zline, y_test, color='red')
ax.scatter3D(xline, zline, y_predict, color='black')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
zline = encoded_sweets[num_training:]
xline = encoded_sex[num_training:]
ax.set_xlabel('sex')
ax.set_zlabel('happiness')
ax.set_ylabel('sweets')
ax.scatter3D(xline, zline, y_test, color='red')
ax.scatter3D(xline, zline, y_predict, color='black')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
zline = encoded_alcohol[num_training:]
xline = encoded_sex[num_training:]
ax.set_xlabel('sex')
ax.set_zlabel('happiness')
ax.set_ylabel('alcohol')
ax.scatter3D(xline, zline, y_test, color='red')
ax.scatter3D(xline, zline, y_predict, color='black')
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
zline = encoded_sex[num_training:]
xline = encoded_cigarettes[num_training:]
ax.set_xlabel('cigarettes')
ax.set_zlabel('happiness')
ax.set_ylabel('sex')
ax.scatter3D(xline, zline, y_test, color='red')
ax.scatter3D(xline, zline, y_predict, color='black')
plt.show()
