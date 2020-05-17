import numpy as np
import matplotlib.pyplot as plt
import math

one = np.ones(15)
noise = np.random.normal(size=20)
x = np.linspace(-3, 3, 20)
x = np.random.permutation(x)


# training error and testing error

testX = x[15:]
X = np.vstack([x[:15],one])

#print("{}".format( noise ))
#print("{}".format( x ))
#print("{}".format( one ))
#print("{}".format( X ))
#print("{}".format( X.shape ))
#print("{}".format( testX ))

y = np.add(x[:15] * 2, noise[:15]) 
#print("{}".format( y ))

testY = np.add(x[15:] * 2, noise[15:]) 
#print("{}".format( testY ))

Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(y.T)
#print("{}".format( Wlin ))

c = x[:15] * Wlin[0] + Wlin[1]
d = x[15:] * Wlin[0] + Wlin[1]
trainError = math.pow( np.square(np.subtract( c, y )).mean(), 1/2 )
print("training error : {}".format( trainError ))
testError = math.pow( np.square(np.subtract( d, testY )).mean(), 1/2 )
print("testing errors : {}".format( testError ))


a = np.linspace(-3, 3, 100)
b = a * Wlin[0] + Wlin[1]

plt.scatter(x[:15], y, color = 'blue')
plt.scatter(x[15:], testY, color = 'yellow')
plt.plot(a, b, color = 'red')
plt.ylim(-8, 8)
plt.title('Fitting plots for training error and testing error')
plt.xlabel("trainingPoint : blue , testingPoint : yellow")
plt.show()



# cross-validation errors ( leave-one-out )

sum = 0

for i in range(15):
	
	tmpX = np.delete(x[:15], i)
	validX = x[i]
	X = np.vstack([tmpX, one[:14]])
	
	#print("{}".format( tmpX ))
	#print("{}".format( validX ))
	#print("{}".format( X ))
	
	tmpY = np.add(tmpX * 2, np.delete(noise[:15], i) ) 
	#print("{}".format( tmpY ))
	
	validY = np.add(validX * 2, noise[i]) 
	#print("{}".format( validY ))
	
	Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(tmpY.T)
	#print("{}".format( Wlin ))
	
	d = validX * Wlin[0] + Wlin[1]
	crossError = np.abs(np.subtract( d, validY ))
	#print("{}".format( crossError ))
	sum += crossError
	
	a = np.linspace(-3, 3, 100)
	b = a * Wlin[0] + Wlin[1]
	plt.plot(a, b, color = 'red')

print("cross-validation errors for leave-one-out : {}".format( sum/15 ))

plt.scatter(x[:15], y, color = 'blue')
plt.ylim(-8, 8)
plt.title('Fitting plots for cross-validation errors ( leave-one-out )')
plt.xlabel("trainingPoint + validationPoint : blue")
plt.show()



# cross-validation errors ( five-fold )

sum = 0

for i in range(0, 15, 3):

	tmpX = np.delete(x[:15], [i, i+1, i+2])
	validX = x[i:i+3]
	X = np.vstack([tmpX, one[:12]])
	
	#print("{}".format( tmpX ))
	#print("{}".format( validX ))
	#print("{}".format( X ))
	
	tmpY = np.add(tmpX * 2, np.delete(noise[:15], [i, i+1, i+2]) ) 
	#print("{}".format( tmpY ))
	
	validY = np.add(validX * 2, noise[i:i+3]) 
	#print("{}".format( validY ))
	
	Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(tmpY.T)
	#print("{}".format( Wlin ))
	
	
	d = validX * Wlin[0] + Wlin[1]
	crossError = math.pow( np.square(np.subtract( d, validY )).mean(), 1/2 )
	#print("{}".format( crossError ))
	sum += crossError
	
	a = np.linspace(-3, 3, 100)
	b = a * Wlin[0] + Wlin[1]
	plt.plot(a, b, color = 'red')


print("cross-validation errors for five-fold : {}".format( sum/5 ))

plt.scatter(x[:15], y, color = 'blue')
plt.ylim(-8, 8)
plt.title('Fitting plots for cross-validation errors ( five-fold )')
plt.xlabel("trainingPoint + validationPoint : blue")
plt.show()
