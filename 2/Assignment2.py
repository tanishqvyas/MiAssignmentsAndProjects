import copy
import time


class PriorityQueue:

    '''
    Insertion at the beginning of list
    Deletion from the back
    '''

    def __init__(self):
        self.pqueue = []
        self.length = 0


    def isQueueEmpty(self):

        if(self.length > 0):
            False
        else:
            True

    def insert(self, value, priority):

        if(self.length == 0):
            self.pqueue.append([value, priority])
        
        else:
            for i in range(len(self.pqueue)):

                if(self.pqueue[i][1] <= priority):
                    self.pqueue.insert(i, [value, priority])
                    break

        self.length += 1
        
    def delete(self):
        
        if(self.length > 0):

            self.pqueue.pop(-1)

            self.length -= 1


    def display(self, showPriority = False):

        # if(showPriority):
        #     for i in range(self.length):
        #         print("cur idx : ", i)
        #         print("Value : ", self.pqueue[i][0], " Priority : ", self.pqueue[i][1])
        
        # else:
        #     for i in range(self.length):
        #         print(self.pqueue[i][0], end=" ")

        # print()     
        print("Len : ", self.length)
        print("Q : ", self.pqueue)



def DFS_Traversal(meta_data, start_point, goals, cost):
    # Initializing the path list
    path = []

    # Variable to keep track whether we found our way or not
    isPathFound = False

    # Visited Array
    visited = [False for i in range(meta_data["node_count"]+1)]

    # Holder stack
    holder_stack = []
    holder_stack.append(start_point)

    while len(holder_stack) > 0:

        node = holder_stack[-1]
        holder_stack.pop()

        if(not visited[node]):
            # Adding to the path
            path.append(node)

            visited[node] = True
            if(node in goals):
                isPathFound = True

        # Checking if the new node is goal node or not
        if(isPathFound):
            break

        # Exploring adjacent nodes
        for adjacent_node in range(1, meta_data["node_count"]+1):
            if(not visited[adjacent_node] and cost[node][adjacent_node] != -1):
                holder_stack.append(adjacent_node)
                break   

    return path


def UCS_Traversal(meta_data, start_point, goals, cost):
    
    # Initialize an empty priority queue
    PriorityQ = PriorityQueue()

    ans_path = []

    # Insert root into queue
    PriorityQ.insert([start_point], 0)

    while not PriorityQ.isQueueEmpty():
        
        print("while loop begins------------------------------")
        PriorityQ.display(showPriority=True)
        time.sleep(1)

        # Dequeue Max Priority element : Which will be the last element in queue with least cost
        path = PriorityQ.pqueue[-1][0]
        priority = PriorityQ.pqueue[-1][1]
        PriorityQ.delete()

        print("Cur path under eval : ", path)

        if(path[-1] in goals):
            ans_path = path
            break

        else:

            # This is the noder to which path has least cost
            ele_to_expand = path[-1]

            # Find all children and add their paths in PQueue
            for i in range(1, meta_data["node_count"]+1):

                if(cost[ele_to_expand][i] != -1):

                    path_to_add = copy.deepcopy(path)
                    path_to_add.append(i)
                    updated_priority = priority + cost[ele_to_expand][i]

                    if(updated_priority > 0 and (i not in path)):
                        PriorityQ.insert(path_to_add, updated_priority)

        PriorityQ.display(showPriority=True)
        print("While loop ends----------------------\n\n")
        time.sleep(1)

    return ans_path


def A_star_Traversal(meta_data, start_point, goals, cost, heuristic):
    path = []

    return path


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
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    
    l = []

    # Getting the meta data for the given graph
    meta_data = {}

    meta_data["node_count"] = len(cost) - 1

    t1 = DFS_Traversal(meta_data, start_point, goals, cost)
    t2 = UCS_Traversal(meta_data, start_point, goals, cost)
    t3 = A_star_Traversal(meta_data, start_point, goals, cost, heuristic)

    l.append(t1)
    l.append(t2)
    l.append(t3)

    return l


if __name__ == '__main__':
    
    obj = PriorityQueue()

    obj.insert("u", 1)
    obj.display()
    obj.insert("o", 2)
    obj.display()
    obj.insert("i", 3)
    obj.display()
    obj.insert("e", 4)
    obj.display()
    obj.insert("a", 4)
    obj.display()
    obj.delete()
    obj.display()
    obj.delete()
    obj.display()
    obj.delete()
    obj.display()
    obj.delete()
    obj.display()
    obj.delete()
    obj.display()
