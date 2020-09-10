'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''



# This function performs DFS traversal on the graph
def Perform_DFS_Traversal(meta_data, start_point, goals, cost):


	# List to hold the path followed while doing DFS for all goal states : [ [path1], [path2], [path3]... ]
	path = []

	for cur_goal in goals:

		# Current path
		cur_path = []

		# visited array
		visited = [0 for j in range(meta_data["node_count"] + 1)]

		# variable to check if goal is reached
		found_goal = False

		# Current stack to store nodes
		holder_stack = []
		holder_stack.append(start_point)
		visited[start_point] = 1

		# While the node stack isnt empty
		while(len(holder_stack) > 0):

			# pop the node and check for children nodes, here insertion in front and deletion from front
			# insertion : l.insert(0, value)
			# deletion : l.pop(0)

			# Popping the node
			popped_node = holder_stack.pop(0)
			
			# Adding the popped node to path
			cur_path.append(popped_node)

			# Traversing in reverse to maintain lexographical order
			for i in range(meta_data["node_count"]-1, 0, -1):

				if(visited[i] == 0 and cost[popped_node][i] >= 0):

					holder_stack.insert(0, i)
					visited[i] = 1

					if(cur_goal == i):
						found_goal = True
						break


			if(found_goal):
				break

		path.append(cur_path)

	print(path)

	return path



# This function performs Uniform Cost Search based traversal
def Perform_Uniform_Cost_Search(meta_data, start_point, goals, cost):
	
	# List to hold the path followed while doing DFS for all goal states : [ [path1], [path2], [path3]... ]
	path = []




	return path	


def Perform_A_Star_Traversal(meta_data, start_point, goals, heuristic, cost):

	# List to hold the path followed while doing DFS for all goal states : [ [path1], [path2], [path3]... ]	
	path = []




	return path

def tri_Traversal(cost, heuristic, start_point, goals):
    
    # final ans list
	l = []

    # Getting the meta data for the given graph
	meta_data = {}

	meta_data["node_count"] = len(cost) - 1


	# Getting answers to all the graph traversals
	print("------------------Doing DFS-------------------")
	t1 = Perform_DFS_Traversal(meta_data, start_point, goals, cost)
	print("------------------Doing UCS-------------------")
	t2 = Perform_Uniform_Cost_Search(meta_data, start_point, goals, cost)
	print("------------------Doing A star-------------------")
	t3 = Perform_A_Star_Traversal(meta_data, start_point, goals, heuristic, cost)

	l.append(t1)
	l.append(t2)
	l.append(t3)

	return l


