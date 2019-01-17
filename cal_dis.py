import numpy as np
from sklearn.preprocessing import normalize 

# using martrix
def min_dis(Q, M, k = 0):
	# find the col pos of min_dis in M
    # Q: 1*d
    # M: p*d
	A = (Q-M)**2 #p*d
	A = np.sum(A, axis=1)
	dis = np.argsort(A)
	if k == 0:
		ret = dis
	else:
		ret = dis[:k]
	#print(ret)
	return ret

# using martrix
def min_angle(Q, M, k = 0):
	# find the row pos of min_angle in M
	Q = normalize(Q, norm='l2')
	M = normalize(M, axis=1, norm='l2')
	A = np.sum(Q*M, axis=1)
	dis = np.argsort(A)
	if k == 0:
		ret = dis
	else:
		ret = dis[-k:]
	#print(ret)
	return ret

def mo_min_angle(Q, M, k = 0, cos = 0.8):
	Q = normalize(Q, norm='l2')
	M = normalize(M, axis=1, norm='l2')
	A = np.sum(Q*M, axis=1)-cos
	A = np.abs(A)
	dis = np.argsort(A)
	if k == 0:
		ret = dis
	else:
		ret = dis[:k]
	#print(ret)
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
def show(pos, y):
	y2 = [y[i, 0] for i in pos]
	print(pos)
	print(y2)

def show_not_nearest(pos, y, k):
	pos2, y2 = [], []
	label_val = y[pos[0], 0]
	pos2 = [i for i in pos if y[i, 0] != label_val]
	y2 = [y[i, 0] for i in pos2]
	print(pos2[:k])
	print(y2[:k])

import numpy as np
import falconn

if __name__ == '__main__':
	a1 = np.load('outputs_1.npy')
	a2 = np.load('outputs_2.npy')
	y = np.load('labels.npy')
	print(y.shape)

	a = np.r_[a1, a2]
	n, d = a.shape

	p = falconn.get_default_parameters(n, d)
	t = falconn.LSHIndex(p)
	dataset = a
	t.setup(dataset)

	Q = t.construct_query_object()
	
	# input	
	i, k = 4545, 100
	print(i, k)
	while(True):
		i, k = map(int, input().split())
		q = a[i:i+1, :]
		u = q.sum(axis=0)
	
		ans = Q.find_k_nearest_neighbors(u, k)
		print(ans)
		y2 = [y[i, 0] for i in ans]
		print(y2)
	
	
	# 1
	print(1)
	pos = mo_min_angle(q, a, k)#most near the cos value of 0.8
	show(pos, y)
	#print(pos)
	#y2 = [y[i,0] for i in pos]
	#print(y2)
	# 2
	print(2)
	pos = min_angle(q, a, k)#most near
	show(pos, y)
	# 3
	print(3)
	pos = min_angle(q, a, 0)
	show_not_nearest(pos, y, 10)
