# This script calculates from any given optical system the parameters for the RQV representative.
import numpy as np
import copy

# Place your operartor sequence related to your optical system here
g_Configuration = [("R", 0),("Q",-1/1.01),("R",10),("Q",-1/1.01),("R",1.01)]


# TODO an operator appearing only once, which is the identity should be removed too

while (g_Configuration[0][0] != "R" and np.count_nonzero(np.array(g_Configuration)[:,0] == "R") > 0) or np.count_nonzero(np.array(g_Configuration)[:,0] == "R") > 1:
	index = len(g_Configuration) - list(np.array(g_Configuration)[::-1,0]).index("R") - 1

	now = copy.copy(g_Configuration[index])
	# We know index > 0
	previous = copy.copy(g_Configuration[index-1])

	# V(b)R(d) == R(d/b^2)V(b)
	if previous[0] == "V":
		g_Configuration[index-1] = ("R", now[1]/previous[1]/previous[1])
		g_Configuration[index] = previous

	# Q(c)R(d) == R(1/(1/d+c)) V(1+cd) Q(1/(1/c+d))
	if previous[0] == "Q":
		g_Configuration[index-1] = ("R", 1/(1/now[1]+previous[1]))
		g_Configuration[index] = ("V", 1 + now[1] * previous[1])
		g_Configuration.insert(index+1, ("Q", 1/(1/previous[1]+now[1])))

	# R(d_1)R(d_2) == R(d_1+d_2)
	if previous[0] == "R":
		g_Configuration[index-1] = ("R", previous[1] + now[1])
		del g_Configuration[index]

	# Remove identity operators
	if g_Configuration[index-1][1] == 0:
		del g_Configuration[index-1]

while (g_Configuration[-1][0] != "V" and np.count_nonzero(np.array(g_Configuration)[:,0] == "V") > 0) or np.count_nonzero(np.array(g_Configuration)[:,0] == "V") > 1:
	index = list(np.array(g_Configuration)[:,0]).index("V")

	now = copy.copy(g_Configuration[index])
	# We know index+1<len(g_Configuration)
	next = copy.copy(g_Configuration[index+1])

	# V(b_1)V(b_2) == V(b_1b_2)
	if next[0] == "V":
		del g_Configuration[index + 1]
		if now[1] * next[1] == 1:
			del g_Configuration[index]
		else:
			g_Configuration[index] = ("V", now[1] * next[1])

	# V(b)Q(c) == Q(b^2c)V(b)
	if next[0] == "Q":
		g_Configuration[index+1] = now
		if now[1]*now[1]*next[1] == 0:
			del g_Configuration[index]
		else:
			g_Configuration[index] = ("Q", now[1]*now[1]*next[1])

	# No need to consider R

while np.count_nonzero(np.array(g_Configuration)[:,0] == "Q") > 1:
	index = list(np.array(g_Configuration)[:,0]).index("Q")

	now = copy.copy(g_Configuration[index])
	# We know index+1<len(g_Configuration)
	next = copy.copy(g_Configuration[index+1])

	# Q(c_1)Q(c_2) == Q(c_1+c_2)
	if next[0] == "Q":
		del g_Configuration[index+1]
		if now[1]+next[1] == 0:
			del g_Configuration[index]
		else:
			g_Configuration[index] = ("Q", now[1]+next[1])

	# No need to consider R or V

print(g_Configuration)
