import numpy as np
import matplotlib.pyplot as plt
import math
import sys

degree = 14
dataPoint = 60

if len(sys.argv)>1 :
	dataPoint = int(sys.argv[1])
print("dataPoint : {}".format( dataPoint ))

trainPoint = int(dataPoint * 3 / 4)
testPoint = int(dataPoint / 4)

one = np.ones(trainPoint)
noise = np.random.normal(size=dataPoint)
x = np.linspace(-3, 3, dataPoint)
x = np.random.permutation(x)
#print("{}".format( x ))


# training error and testing error

testX = x[trainPoint:]
X = np.vstack([ np.power(x[:trainPoint], degree), np.power(x[:trainPoint], degree-1) ])
for i in range(degree-2):
	X = np.vstack([ X, np.power(x[:trainPoint], degree-2-i) ])
X = np.vstack([ X, one])

#print("{}".format( noise ))
#print("{}".format( x ))
#print("{}".format( one ))
#print("{}".format( X ))
#print("{}".format( X.shape ))
#print("{}".format( testX ))

y = np.add(x[:trainPoint] * 2, noise[:trainPoint]) 
#print("{}".format( y ))

testY = np.add(x[trainPoint:] * 2, noise[trainPoint:]) 
#print("{}".format( testY ))

Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(y.T)
#print("{}".format( Wlin ))


c = 0
for i in range(degree):
	c += np.power(x[:trainPoint], degree-i) * Wlin[i]
c += Wlin[degree]

d = 0
for i in range(degree):
	d += np.power(x[trainPoint:], degree-i) * Wlin[i]
d += Wlin[degree]

trainError = math.pow( np.square(np.subtract( c, y )).mean(), 1/2 )
print("training error : {}".format( trainError ))
testError = math.pow( np.square(np.subtract( d, testY )).mean(), 1/2 )
print("testing errors : {}".format( testError ))


a = np.linspace(-3, 3, 100)
b = 0
for i in range(degree):
	b += np.power(a, degree-i) * Wlin[i]
b += Wlin[degree]
#b = np.power(a, 5) * Wlin[0] + np.power(a, 4) * Wlin[1] + np.power(a, 3) * Wlin[2] + np.power(a, 2) * Wlin[3] + a * Wlin[4] + Wlin[5]

plt.scatter(x[:trainPoint], y, color = 'blue')
plt.scatter(x[trainPoint:], testY, color = 'yellow')
plt.plot(a, b, color = 'red')
plt.ylim(-8, 8)
plt.title('Fitting plots for training error and testing error\nwith degree - {D} and dataPoint {P}'.format( D=degree, P=dataPoint))
plt.xlabel("trainingPoint : blue , testingPoint : yellow")
plt.show()



# cross-validation errors ( five-fold )

sum = 0

for i in range(0, trainPoint, int(trainPoint/5)):

	deleted = []
	for j in range(int(trainPoint/5)):
		deleted.append(i+j)
	tmpX = np.delete(x[:trainPoint], deleted)
	validX = x[i:i+int(trainPoint/5)]
	X = np.vstack([ np.power(tmpX, degree), np.power(tmpX, degree-1) ])
	for j in range(degree-2):
		X = np.vstack([ X, np.power(tmpX, degree-2-j) ])
	cnt = trainPoint-int(trainPoint/5)
	X = np.vstack([ X, one[:cnt]])
	
	#print("{}".format( tmpX ))
	#print("{}".format( validX ))
	#print("{}".format( X ))
	
	tmpY = np.add(tmpX * 2, np.delete(noise[:trainPoint], deleted) ) 
	#print("{}".format( tmpY ))
	
	validY = np.add(validX * 2, noise[i:i+int(trainPoint/5)]) 
	#print("{}".format( validY ))
	
	Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(tmpY.T)
	#print("{}".format( Wlin ))
	
	
	d = 0
	for j in range(degree):
		d += np.power(validX, degree-j) * Wlin[j]
	d += Wlin[degree]
	crossError = math.pow( np.square(np.subtract( d, validY )).mean(), 1/2 )
	#print("{}".format( crossError ))
	sum += crossError
	
	a = np.linspace(-3, 3, 100)
	b = 0
	for i in range(degree):
		b += np.power(a, degree-i) * Wlin[i]
	b += Wlin[degree]
	plt.plot(a, b, color = 'red')

print("cross-validation errors for five-fold : {}".format( sum/5 ))

plt.scatter(x[:trainPoint], y, color = 'blue')
plt.ylim(-8, 8)
plt.title('Fitting plots for cross-validation errors ( five-fold )\nwith degree - {D} and dataPoint {P}'.format( D=degree, P=dataPoint))
plt.xlabel("trainingPoint + validationPoint : blue")
plt.show()