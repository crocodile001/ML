import numpy as np
import matplotlib.pyplot as plt
import math

color = ['green', 'darkred', 'skyblue']
degree = [5, 10, 14]


one = np.ones(15)
noise = np.random.normal(size=20)
x = np.linspace(-3, 3, 20)
x = np.random.permutation(x)
#print("{}".format( x ))


# training error and testing error

for k in range(len(degree)):

	testX = x[15:]
	X = np.vstack([ np.power(x[:15], degree[k]), np.power(x[:15], degree[k]-1) ])
	for i in range(degree[k]-2):
		X = np.vstack([ X, np.power(x[:15], degree[k]-2-i) ])
	X = np.vstack([ X, one])

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


	c = 0
	for i in range(degree[k]):
		c += np.power(x[:15], degree[k]-i) * Wlin[i]
	c += Wlin[degree[k]]

	d = 0
	for i in range(degree[k]):
		d += np.power(x[15:], degree[k]-i) * Wlin[i]
	d += Wlin[degree[k]]

	trainError = math.pow( np.square(np.subtract( c, y )).mean(), 1/2 )
	print("training error with degree {D} : {T}".format( D=degree[k],T=trainError ))
	testError = math.pow( np.square(np.subtract( d, testY )).mean(), 1/2 )
	print("testing errors with degree {D} : {T}".format( D=degree[k],T=testError ))


	a = np.linspace(-3, 3, 100)
	b = 0
	for i in range(degree[k]):
		b += np.power(a, degree[k]-i) * Wlin[i]
	b += Wlin[degree[k]]
	#b = np.power(a, 5) * Wlin[0] + np.power(a, 4) * Wlin[1] + np.power(a, 3) * Wlin[2] + np.power(a, 2) * Wlin[3] + a * Wlin[4] + Wlin[5]

	plt.plot(a, b, color = color[k])

plt.scatter(x[:15], y, color = 'blue')
plt.scatter(x[15:], testY, color = 'yellow')
plt.ylim(-8, 8)
plt.title('Fitting plots for training error and testing error with degree - {}'.format(degree))
plt.xlabel("trainingPoint : blue , testingPoint : yellow\ngreen : degree-5 , darkred : degree-10 , skyblue : degree-14")
plt.show()



# cross-validation errors ( leave-one-out )

for k in range(len(degree)):

	sum = 0

	for i in range(15):
		
		tmpX = np.delete(x[:15], i)
		validX = x[i]
		X = np.vstack([ np.power(tmpX, degree[k]), np.power(tmpX, degree[k]-1) ])
		for j in range(degree[k]-2):
			X = np.vstack([ X, np.power(tmpX, degree[k]-2-j) ])
		X = np.vstack([ X, one[:14]])
		
		#print("{}".format( tmpX ))
		#print("{}".format( validX ))
		#print("{}".format( X ))
		
		tmpY = np.add(tmpX * 2, np.delete(noise[:15], i) ) 
		#print("{}".format( tmpY ))
		
		validY = np.add(validX * 2, noise[i]) 
		#print("{}".format( validY ))
		
		Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(tmpY.T)
		#print("{}".format( Wlin ))
		
		d = 0
		for j in range(degree[k]):
			d += np.power(validX, degree[k]-j) * Wlin[j]
		d += Wlin[degree[k]]
		crossError = np.abs(np.subtract( d, validY ))
		#print("{}".format( crossError ))
		sum += crossError
		
		a = np.linspace(-3, 3, 100)
		b = 0
		for j in range(degree[k]):
			b += np.power(a, degree[k]-j) * Wlin[j]
		b += Wlin[degree[k]]
		plt.plot(a, b, color = 'red')

	print("cross-validation errors for leave-one-out with degree {D} : {S}".format( D=degree[k], S=sum/15 ))

	plt.scatter(x[:15], y, color = 'blue')
	plt.ylim(-8, 8)
	plt.title('Fitting plots for cross-validation errors ( leave-one-out ) with degree - {}'.format(degree[k]))
	plt.xlabel("trainingPoint + validationPoint : blue")
	plt.show()



# cross-validation errors ( five-fold )

for k in range(len(degree)):

	sum = 0

	for i in range(0, 15, 3):

		tmpX = np.delete(x[:15], [i, i+1, i+2])
		validX = x[i:i+3]
		X = np.vstack([ np.power(tmpX, degree[k]), np.power(tmpX, degree[k]-1) ])
		for j in range(degree[k]-2):
			X = np.vstack([ X, np.power(tmpX, degree[k]-2-j) ])
		X = np.vstack([ X, one[:12]])
		
		#print("{}".format( tmpX ))
		#print("{}".format( validX ))
		#print("{}".format( X ))
		
		tmpY = np.add(tmpX * 2, np.delete(noise[:15], [i, i+1, i+2]) ) 
		#print("{}".format( tmpY ))
		
		validY = np.add(validX * 2, noise[i:i+3]) 
		#print("{}".format( validY ))
		
		Wlin = np.linalg.inv(X.dot(X.T)).dot(X).dot(tmpY.T)
		#print("{}".format( Wlin ))
		
		
		d = 0
		for j in range(degree[k]):
			d += np.power(validX, degree[k]-j) * Wlin[j]
		d += Wlin[degree[k]]
		crossError = math.pow( np.square(np.subtract( d, validY )).mean(), 1/2 )
		#print("{}".format( crossError ))
		sum += crossError
		
		a = np.linspace(-3, 3, 100)
		b = 0
		for j in range(degree[k]):
			b += np.power(a, degree[k]-j) * Wlin[j]
		b += Wlin[degree[k]]
		plt.plot(a, b, color = 'red')

	print("cross-validation errors for five-fold with degree {D} : {S}".format( D=degree[k], S=sum/5 ))

	plt.scatter(x[:15], y, color = 'blue')
	plt.ylim(-8, 8)
	plt.title('Fitting plots for cross-validation errors ( five-fold ) with degree - {}'.format(degree[k]))
	plt.xlabel("trainingPoint + validationPoint : blue")
	plt.show()
