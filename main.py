
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow

df = pd.read_csv("C:\\Users\\Suvarna Lakshmi\\PycharmProjects\\compressiveStrengthPrediction\\Concrete_Data_Yeh.csv")
x_org = df.drop('csMPa',axis=1).values
y_org = df['csMPa'].values


corr = df.corr()
sns.heatmap(corr,xticklabels=True,yticklabels=True,annot = True,cmap ='coolwarm')
plt.title("Correlation Between Variables")
plt.savefig('1.png')

# # pair Plot
sns.pairplot(df,palette="husl",diag_kind="kde")
plt.savefig('2.png')

# Using Test/Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_org,y_org, test_size=0.3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building ANN As a Regressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam

#Defining Root Mean Square Error As our Metric Function
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#Building  first layer Layers
model=Sequential()

model.add(Dense(64,input_dim=8,activation = 'relu'))

# Bulding Second and third layer
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())

# Output Layer
model.add(Dense(1,activation='linear'))

# Optimize , Compile And Train The Model
opt =Adam(lr=0.0015)

model.compile(optimizer=opt,loss='mse',metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train,y_train,epochs = 35 ,batch_size=32,validation_split=0.1)

print(model.summary())

y_predict = model.predict(X_test)
pr=model.predict([22,22,22,22,22,22,22,22])

from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))
model.save("compressive.h5", save_format="h5")


# Plotting Loss And Root Mean Square Error For both Training And Test Sets
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Root Mean Squared Error')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('4.png')
plt.show()

model.save("compressive.model", save_format="h5")



