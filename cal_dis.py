import numpy as np
from sklearn.preprocessing import normalize 

# using martrix
def min_dis(Q, M, k):
	# find the col pos of min_dis in M
    # Q: 1*d
    # M: p*d
	A = (Q-M)**2 #p*d
	A = np.sum(A, axis=1)
	dis = np.argsort(A)
	ret = dis[:k]
	print(ret)
	return ret

# using martrix
def min_angle(Q, M, k):
	# find the row pos of min_angle in M
	Q = normalize(Q, norm='l2')
	M = normalize(M, axis=1, norm='l2')

	A = np.sum(Q*M, axis=1)
	dis = np.argsort(A)
	ret = dis[-k:]
	print(ret)
	return ret

def mo_min_angle(Q, M, k):
	Q = normalize(Q, norm='l2')
	M = normalize(M, axis=1, norm='l2')
	A = np.sum(Q*M, axis=1)-0.8
	A = np.abs(A)
	dis = np.argsort(A)
	ret = dis[k:k+k]
	print(ret)
	return ret



a = np.array([[4,2,3]])
b = np.array([[0,1,1],[1,8,1],[2,1,0]])
# print(a-b)
# print(a[:,-2:])
# print(b)
# print('#')
# min_dis(a, b, 2)
# min_angle(a, b, 2)
# print('#')
# print(normalize(a, norm = 'l2'))
# print(normalize(b, axis = 0, norm = 'l2'))


#p,d = 1300000,350
#a = np.random.rand(p,d)
#b = np.random.rand(1,d)
#min_angle(b, a, 20)
#min_dis(b, a, 20)
#min_angle(b, a, 20)
import numpy as np
if __name__ == '__main__':
	a1 = np.load('outputs_1.npy')
	a2 = np.load('outputs_2.npy')
	y = np.load('labels.npy')
	print(y.shape)

	print(y.shape)
	a = np.r_[a1, a2]
	i = 50012
	q = a[i:i+1, :]
	pos = mo_min_angle(q, a, 30)
	print(pos)
	y2 = [y[i,0] for i in pos]
	print(y2)
