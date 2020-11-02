from Assignment2 import *
def test1():
    '''size of cost matrix is 11x11
    0th row and 0th column is ALWAYS 0
    Number of nodes is 10
    size of heuristic list is 11
    0th index is always 0'''

    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]

    try:
        # start node is 1, goal states to reach are 6,7 & 10
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[0] == [1, 2, 3, 4, 7]:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")
    try:
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[1] == [1, 5, 4, 7]:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE UCS_TRAVERSAL FAILED")

    try:
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[2] == [1, 5, 4, 7]:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3 FOR THE A_star_TRAVERSAL FAILED")
    print("-----------------------------------------------------")
    print("1",tri_traversal(cost,heuristic, 8, [6,7, 10])==[[8, 5, 4, 1, 2, 6], [8, 10], [8, 10]])
    print("2",tri_traversal(cost,heuristic, 3, [6, 7, 10])==[[3, 2, 6], [3, 4, 7], [3, 4, 7]])
    print("3",tri_traversal(cost,heuristic, 8, [8])==[[8], [8], [8]])
    # print("4",tri_traversal(cost,heuristic, 9, [6])==[[], [], []])
    print("5",tri_traversal(cost,heuristic, 8, [10])==[[8, 5, 9, 10], [8, 10], [8, 10]])
    print("6",tri_traversal(cost,heuristic, 4, [6, 7, 10])==[[4, 1, 2, 6], [4, 7], [4, 7]])

def test2():
    cost = [[0, 0, 0, 0],
            [0, 0, 5,10] ,
            [0, -1, 0, 5],
            [0, -1, -1, 0]]
    heuristic = [0, 0, 0, 0]

    print("7",tri_traversal(cost,heuristic, 1, [3])==[[1,2,3],[1,2,3],[1,2,3]])

def test_case1():
    cost = [[0, 0, 0, 0,0],
            [0, 0, -1, 10,5],
            [0, -1, 0, -1,-1],
            [0, -1, -1, 0,-1],
            [0, -1, -1, 5,0]]
    heuristic = [0, 0, 0, 0,0]

    print("8",tri_traversal(cost,heuristic, 1, [3])==[[1,3],[1,3],[1,3]])

def test_case2():
    cost = [[0, 0, 0, 0],
            [0, 0, -1, 10],
            [0, -1, 0, -1],
            [0, -1, -1, 0]]
    heuristic = [0, 0, 0, 0]

    print("9",tri_traversal(cost,heuristic, 1, [3])==[[1,3],[1,3],[1,3]])
    print("10",tri_traversal(cost,heuristic, 3, [3])==[[3],[3],[3]])

def test_case3():
    cost = [[0, 0, 0, 0, 0],
            [0, 0, -1, -1,10],
            [0, -1, 0, 5,-1],
            [0, -1, 6, 0,-1],
            [0, -1, -1, 0,8]]
    heuristic = [0, 0, 0, 0, 0]

    print("11",tri_traversal(cost,heuristic, 1, [4])==[[1, 4], [1, 4], [1,4]])
    print("12",tri_traversal(cost,heuristic, 2, [3,4])==[[2, 3], [2, 3], [2, 3]])


def test_case4():
    cost = [[0,0,0,0,0,0,0],
            [0,0,2,0,0,10,7],
            [0,0,0,3,0,0,0],
            [0,0,0,0,2,0,2],
            [0,0,0,0,0,3,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,3,0],
            ]
    if (tri_traversal(cost,[0,0,0,0,0,0,0],1,[5])[1]==[1,2,3,4,5]):
        print("SAMPLE TEST CASE 13 PASSED")
    else:
        print("SAMPLE TEST CASE 13 FAILED")

def test_case5():
    cost = [[0,0,0,0,0,0,0],
            [0,0,2,-1,-1,10,-1],
            [0,-1,0,2,-1,-1,-1],
            [0,-1,-1,0,2,-1,-1],
            [0,-1,-1,-1,0,-1,2],
            [0,-1,-1,-1,-1,0,-1],
            [0,-1,-1,-1,-1,2,0]
            ]
    if (tri_traversal(cost,[0,0,0,0,0,0,0],1,[5])[1]==[1,2,3,4,6,5]):
        print("SAMPLE TEST CASE 14 PASSED")
    else:
        print("SAMPLE TEST CASE 14 FAILED")
    print("-----------------------------------------------------")

test1()
test2()
test_case1()
test_case2()
test_case3()
test_case4()
test_case5()

def test_case():
    """size of cost matrix is 11x11
    0th row and 0th column is ALWAYS 0
    Number of nodes is 10
    size of heuristic list is 11
    0th index is always 0"""

    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    cost2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 7, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 16],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    cost3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 16],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    cost4 = [[0,0,0,0,0,0,0],
            [0,0,1,-1,-1,-1,-1],
            [0,-1,0,3,-1,-1,6],
            [0,-1,-1,0,4,-1,-1],
            [0,-1,-1,-1,0,5,-1],
            [0,-1,5,-1,-1,0,-1],
            [0,-1,-1,-1,-1,-1,0]]
    
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
    heuristic2 = [0, 5, 7, 3, 4, 6, 0]

    try:
        # start node is 1, goal states to reach are 6,7 & 10
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[0] == [1, 2, 3, 4, 7]:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")
    try:
        #print((tri_traversal(cost,heuristic, 8, [6, 7, 10]))[0])
        if ((tri_traversal(cost,heuristic, 8, [6, 7, 10]))[0] == [8, 5, 4, 1, 2, 6]):
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")
    try:
        # start node is 1, goal states to reach are 6,7 & 10
        if (tri_traversal(cost,heuristic, 1, [10,6]))[0] == [1, 2, 3, 4 , 8 , 5 , 9 ,10]:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")
    try:
        if (tri_traversal(cost,heuristic, 1, [6, 7, 10]))[1] == [1, 5, 4, 7]:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE UCS_TRAVERSAL FAILED")
    try:
        #print((tri_traversal(cost2,heuristic, 1, [6, 7, 10]))[1])
        if (tri_traversal(cost2,heuristic, 1, [6, 7, 10]))[1] == [1, 3, 4, 7]:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE UCS_TRAVERSAL FAILED")
    try:
        #print((tri_traversal(cost2,heuristic, 1, [10]))[1])
        if (tri_traversal(cost,heuristic, 1, [10]))[2] == [1, 5, 9, 10]:
            print("SAMPLE TEST CASE 2 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE A_star_TRAVERSAL FAILED")
    try:
        #print((tri_traversal(cost2,heuristic, 1, [10]))[2])
        if (tri_traversal(cost2,heuristic, 1, [10]))[2] == [1,3,4,8,10]:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3 FOR THE A_star_TRAVERSAL FAILED")
    try:
        if (tri_traversal(cost3,heuristic, 1, [10]))[2] == [1,5,4,8,10]:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3 FOR THE A_star_TRAVERSAL FAILED")
    
    try:
        if (tri_traversal(cost4, heuristic2, 1, [6]))[0] == [1,2,6]:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 1 FOR THE  DFS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 1 FOR THE DFS_TRAVERSAL FAILED")
    try:
        if (tri_traversal(cost4, heuristic2, 1, [6]))[1] == [1,2,6]:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 2 FOR THE  UCS_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 2 FOR THE UCS_TRAVERSAL FAILED")
    try:
        if (tri_traversal(cost4, heuristic2, 1, [6]))[2] == [1,2,6]:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL PASSED")
        else:
            print("SAMPLE TEST CASE 3 FOR THE  A_star_TRAVERSAL FAILED")
    except:
        print("SAMPLE TEST CASE 3 FOR THE A_star_TRAVERSAL FAILED")

test_case()


cost1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]

cost2 = [[0,0,0,0,0,0,0,0],
	[0,0,3,-1,-1,-1,-1,2],
	[0,-1,0,5,10,-1,-1,-1],
	[0,-1,-1,0,2,-1,1,-1],
	[0,-1,-1,-1,0,4,-1,-1],
	[0,-1,-1,-1,-1,0,-1,-1],
	[0,-1,-1,-1,-1,3,0,-1],
	[0,-1,-1,1,-1,-1,4,0]] #https://www.geeksforgeeks.org/search-algorithms-in-ai/

cost3 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 6, -1, -1, -1],
    [0, 6, 0, 3, 3, -1],
    [0, -1, 3, 0, 1, 7],
    [0, -1, 3, 1, 0, 8],
    [0, -1, -1, 7, 8, 0],
]

heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
heuristic2 = [0,7,9,4,2,0,3,5]
heuristic3 = [0, 10, 8, 7, 7, 3]


def dfstest():
    print("---------------------------------- TESTS FOR DFS SEARCH----------------------------------------")
    print("Test 1: ", tri_traversal(cost1,heuristic1, 1, [1])[0] == [1])
    print("Test 2: ", tri_traversal(cost1,heuristic1, 1, [2])[0] == [1,2])
    print("Test 3: ", tri_traversal(cost1,heuristic1, 1, [3])[0] == [1,2,3])
    print("Test 4: ", tri_traversal(cost1,heuristic1, 1, [4])[0] == [1,2,3,4])
    print("Test 5: ", tri_traversal(cost1,heuristic1, 1, [5])[0] == [1,2,3,4,8,5])
    print("Test 6: ", tri_traversal(cost1,heuristic1, 1, [6])[0] == [1,2,6])
    print("Test 7: ", tri_traversal(cost1,heuristic1, 1, [7])[0] == [1,2,3,4,7])
    print("Test 8: ", tri_traversal(cost1,heuristic1, 1, [8])[0] == [1,2,3,4,8])
    print("Test 9: ", tri_traversal(cost1,heuristic1, 1, [9])[0] == [1,2,3,4,8,5,9])
    print("Test 10: ", tri_traversal(cost1,heuristic1, 1, [10])[0] == [1, 2, 3, 4, 8, 5, 9, 10])
    print("Test 11: ", tri_traversal(cost1,heuristic1, 1, [6,7,10])[0] == [1,2,3,4,7])
    print("Test 12: ", tri_traversal(cost1,heuristic1, 1, [3,4,7,10])[0] == [1,2,3])
    print("Test 13: ", tri_traversal(cost1,heuristic1, 1, [5,9,4])[0] == [1,2,3,4])
    print("Test 14: ", tri_traversal(cost1,heuristic1, 1, [4,8,10])[0]== [1,2,3,4])
    print("Test 15: ", tri_traversal(cost1,heuristic1, 1, [2,8,5])[0] == [1,2])
    print("Test 16: ", tri_traversal(cost1,heuristic1, 1, [7,9,10])[0] == [1,2,3,4,7])
    print("Test 17: ", tri_traversal(cost1,heuristic1, 1, [10,6,8,4])[0] == [1,2,3,4])
    print("Test 18: ", tri_traversal(cost1,heuristic1, 1, [9,7,5,10])[0] == [1,2,3,4,7])
    print("Test 19: ", tri_traversal(cost2,heuristic2, 1, [1])[0] == [1])
    print("Test 20: ", tri_traversal(cost2,heuristic2, 1, [2])[0] == [1,2])
    print("Test 21: ", tri_traversal(cost2,heuristic2, 1, [3])[0] == [1,2,3])
    print("Test 22: ", tri_traversal(cost2,heuristic2, 1, [4])[0] == [1,2,3,4])
    print("Test 23: ", tri_traversal(cost2,heuristic2, 1, [5])[0] == [1,2,3,4,5])
    print("Test 24: ", tri_traversal(cost2,heuristic2, 1, [6])[0] == [1,2,3,6])
    print("Test 25: ", tri_traversal(cost2,heuristic2, 1, [7])[0] == [1,7])
    print("Test 26: ", tri_traversal(cost2,heuristic2, 1, [4,5,6])[0] == [1,2,3,4])
    print("Test 27: ", tri_traversal(cost2,heuristic2, 1, [3,6,7])[0] == [1,2,3])
    print("Test 28: ", tri_traversal(cost2,heuristic2, 1, [4,6])[0] == [1,2,3,4])
    print("Test 29: ", tri_traversal(cost2,heuristic2, 1, [2,3,7])[0] == [1,2])

    # print("Test 30: ", tri_traversal(cost2,heuristic2, 4, [3])[0] == [])
    print("Test 31: ", tri_traversal(cost3,heuristic3, 1, [5])[0] == [1,2,3,4,5])


def ucstest():        
    print("----------------------------------TESTS FOR UCS SEARCH----------------------------------------")
    print("Test 1: ", tri_traversal(cost1,heuristic1,1, [1])[1]== [1])
    print("Test 2: ", tri_traversal(cost1,heuristic1,1, [2])[1]== [1, 2])
    print("Test 3: ", tri_traversal(cost1,heuristic1,1, [3])[1]==[1,2,3])
    print("Test 4: ", tri_traversal(cost1,heuristic1,1, [4])[1]== [1, 5, 4])
    print("Test 5: ", tri_traversal(cost1,heuristic1,1, [5])[1]==[1,5])
    print("Test 6: ", tri_traversal(cost1,heuristic1,1, [6])[1]==[1,2,6])
    print("Test 7: ", tri_traversal(cost1,heuristic1,1, [7])[1]==[1,5,4,7])
    print("Test 8: ", tri_traversal(cost1,heuristic1,1, [8])[1]==[1,5,4,8])
    print("Test 9: ", tri_traversal(cost1,heuristic1,1, [9])[1]==[1,5,9])
    print("Test 10: ", tri_traversal(cost1,heuristic1,1, [10])[1]==[1,5,9,10])
    print("Test 11: ", tri_traversal(cost1,heuristic1,1, [6,7,10])[1]==[1,5,4,7])
    print("Test 12: ", tri_traversal(cost1,heuristic1,1, [3,4,7,10])[1]==[1,2,3])
    print("Test 13: ", tri_traversal(cost1,heuristic1,1, [5,9,4])[1]==[1,5])
    print("Test 14: ", tri_traversal(cost1,heuristic1,1, [4,8,10])[1]==[1,5,4])
    print("Test 15: ", tri_traversal(cost1,heuristic1,1, [2,8,5])[1]==[1,2])
    print("Test 16: ", tri_traversal(cost1,heuristic1,1, [7,9,10])[1]==[1,5,9])
    print("Test 17: ", tri_traversal(cost1,heuristic1,1, [10,6,8,4])[1]==[1,5,4])
    print("Test 18: ", tri_traversal(cost1,heuristic1,1, [9,7,5,10])[1]==[1,5])


def astartest():
    print("----------------------------------TESTS FOR A* SEARCH----------------------------------------")
    print("Test 1: ", tri_traversal(cost1, heuristic1, 1, [1])[2]==[1])
    print("Test 2: ", tri_traversal(cost1, heuristic1, 1, [2])[2]==[1,2])
    print("Test 3: ", tri_traversal(cost1, heuristic1, 1, [3])[2]==[1,2,3])
    print("Test 4: ", tri_traversal(cost1, heuristic1, 1, [4])[2]==[1,5,4])
    print("Test 5: ", tri_traversal(cost1, heuristic1, 1, [5])[2]==[1,5])
    print("Test 6: ", tri_traversal(cost1, heuristic1, 1, [6])[2]==[1,2,6])
    print("Test 7: ", tri_traversal(cost1, heuristic1, 1, [7])[2]==[1,5,4,7])
    print("Test 8: ", tri_traversal(cost1, heuristic1, 1, [8])[2]==[1,5,4,8])
    print("Test 9: ", tri_traversal(cost1, heuristic1, 1, [9])[2]==[1,5,9])
    print("Test 10: ", tri_traversal(cost1, heuristic1, 1, [10])[2]==[1,5,9,10])
    print("Test 11: ", tri_traversal(cost1, heuristic1, 1, [6,7,10])[2]==[1,5,4,7])
    print("Test 12: ", tri_traversal(cost1, heuristic1, 1, [3,4,7,10])[2]==[1,2,3])
    print("Test 13: ", tri_traversal(cost1, heuristic1, 1, [5,9,4])[2]==[1,5])
    print("Test 14: ", tri_traversal(cost1, heuristic1, 1, [4,8,10])[2]==[1,5,4])
    print("Test 15: ", tri_traversal(cost1, heuristic1, 1, [2,8,5])[2]==[1,2])
    print("Test 16: ", tri_traversal(cost1, heuristic1, 1, [7,9,10])[2]==[1,5,4,7])
    print("Test 17: ", tri_traversal(cost1, heuristic1, 1, [10,6,8,4])[2]==[1,5,4])
    print("Test 18: ", tri_traversal(cost1, heuristic1, 1, [9,7,5,10])[2]==[1,5])
    print("Test 19: ", tri_traversal(cost2, heuristic2, 1, [1])[2]==[1])
    print("Test 20: ", tri_traversal(cost2, heuristic2, 1, [2])[2]==[1,2])
    print("Test 21: ", tri_traversal(cost2, heuristic2, 1, [3])[2]==[1,7,3])
    print("Test 22: ", tri_traversal(cost2, heuristic2, 1, [4])[2]==[1,7,3,4])
    print("Test 23: ", tri_traversal(cost2, heuristic2, 1, [5])[2]==[1,7,3,6,5])
    print("Test 24: ", tri_traversal(cost2, heuristic2, 1, [6])[2]==[1,7,3,6])
    print("Test 25: ", tri_traversal(cost2, heuristic2, 1, [7])[2]==[1,7])
    print("Test 26: ", tri_traversal(cost2, heuristic2, 1, [4,5,6])[2]==[1,7,3,4])
    print("Test 27: ", tri_traversal(cost2, heuristic2, 1, [3,6,7])[2]==[1,7])
    print("Test 28: ", tri_traversal(cost2, heuristic2, 1, [4,6])[2]==[1,7,3,4])
    print("Test 29: ", tri_traversal(cost2, heuristic2, 1, [2,3,7])[2]==[1,7])
    # print("Test 30: ", tri_traversal(cost2, heuristic2, 4, [3])[2]==[])
    print("Test 31: ", tri_traversal(cost3, heuristic3, 1, [5])[2]==[1,2,3,5])

#dfstestcheck() #uncomment if you want to check what ur code prints
dfstest()	
astartest()	
ucstest()
print("-------------------------------------------------------")

def test(cal, exp, case ):
    ok = {0: "DFS", 1: "UCS", 2: "A*S"}
    print("Test Case : ",case)
    for i in range(3):
        if(exp[i] == cal[i]):
            print("{0} : PASS".format(ok[i]))
        else:
            print("{0} : FAIL ---- Expected: {1}  Got: {2}".format(ok[i],exp[i],cal[i]))
    print()		
cost1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
        
cost2 = [[0,0,0,0,0,0,0,0],
	[0,0,3,-1,-1,-1,-1,2],
	[0,-1,0,5,10,-1,-1,-1],
	[0,-1,-1,0,2,-1,1,-1],
	[0,-1,-1,-1,0,4,-1,-1],
	[0,-1,-1,-1,-1,0,-1,-1],
	[0,-1,-1,-1,-1,3,0,-1],
	[0,-1,-1,1,-1,-1,4,0]] #https://www.geeksforgeeks.org/search-algorithms-in-ai/

cost3 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 6, -1, -1, -1],
    [0, 6, 0, 3, 3, -1],
    [0, -1, 3, 0, 1, 7],
    [0, -1, 3, 1, 0, 8],
    [0, -1, -1, 7, 8, 0],
]

heuristic1 = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
heuristic2 = [0,7,9,4,2,0,3,5]
heuristic3 = [0, 10, 8, 7, 7, 3]


x = tri_traversal(cost1, heuristic1, 1, [1])
test(x,[[1],[1],[1]], 1)


x = tri_traversal(cost1, heuristic1, 1, [2])
test(x,[[1,2],[1,2],[1,2]], 2)


x = tri_traversal(cost1, heuristic1, 1, [3])
test(x,[[1,2,3],[1,2,3],[1,2,3]], 3)


x = tri_traversal(cost1, heuristic1, 1, [4])
test(x,[[1,2,3,4],[1,5,4],[1,5,4]], 4)


x = tri_traversal(cost1, heuristic1, 1, [5])
test(x,[[1,2,3,4,8,5],[1,5],[1,5]], 5)


x = tri_traversal(cost1, heuristic1, 1, [6])
test(x,[[1,2,6],[1,2,6],[1,2,6]], 6)


x = tri_traversal(cost1, heuristic1, 1, [7])
test(x,[[1,2,3,4,7],[1,5,4,7],[1,5,4,7]], 7)


x = tri_traversal(cost1, heuristic1, 1, [8])
test(x,[[1,2,3,4,8],[1,5,4,8],[1,5,4,8]], 8)


x = tri_traversal(cost1, heuristic1, 1, [9])
test(x,[[1,2,3,4,8,5,9],[1,5,9],[1,5,9]], 9)


x = tri_traversal(cost1, heuristic1, 1, [10])
test(x,[[1,2,3,4,8,5,9,10],[1,5,9,10],[1,5,9,10]], 10)


x = tri_traversal(cost1, heuristic1, 1, [6,7,10])
test(x,[[1,2,3,4,7],[1,5,4,7],[1,5,4,7]], 11)


x = tri_traversal(cost1, heuristic1, 1, [3,4,7,10])
test(x,[[1,2,3],[1,2,3],[1,2,3]], 12)


x = tri_traversal(cost1, heuristic1, 1, [5,9,4])
test(x,[[1,2,3,4],[1,5],[1,5]], 13)


x = tri_traversal(cost1, heuristic1, 1, [4,8,10])
test(x,[[1,2,3,4],[1,5,4],[1,5,4]], 14)


x = tri_traversal(cost1, heuristic1, 1, [2,8,5])
test(x,[[1,2],[1,2],[1,2]], 15)


x = tri_traversal(cost1, heuristic1, 1, [7,9,10])
test(x,[[1,2,3,4,7],[1,5,9],[1,5,4,7]], 16) # a* != dfs here


x = tri_traversal(cost1, heuristic1, 1, [10,6,8,4])
test(x,[[1,2,3,4],[1,5,4],[1,5,4]], 17)


x = tri_traversal(cost1, heuristic1, 1, [9,7,5,10])
test(x,[[1,2,3,4,7],[1,5],[1,5]], 18)


x = tri_traversal(cost2, heuristic2, 1, [1])
test(x,[[1],[1],[1]], 19)


x = tri_traversal(cost2, heuristic2, 1, [2])
test(x,[[1,2],[1,2],[1,2]], 20)


x = tri_traversal(cost2, heuristic2, 1, [3])
test(x,[[1,2,3],[1,7,3],[1,7,3]], 21)


x = tri_traversal(cost2, heuristic2, 1, [4])
test(x,[[1,2,3,4],[1,7,3,4],[1,7,3,4]], 22)


x = tri_traversal(cost2, heuristic2, 1, [5])
test(x,[[1,2,3,4,5],[1,7,3,6,5],[1,7,3,6,5]], 23)


x = tri_traversal(cost2, heuristic2, 1, [6])
test(x,[[1,2,3,6],[1,7,3,6],[1,7,3,6]], 24)


x = tri_traversal(cost2, heuristic2, 1, [7])
test(x,[[1,7],[1,7],[1,7]], 25)


x = tri_traversal(cost2, heuristic2, 1, [4,5,6])
test(x,[[1,2,3,4],[1,7,3,6],[1,7,3,4]], 26)# i donno a* can be = [1,7,3,6] cuz it costs the same f(n)


x = tri_traversal(cost2, heuristic2, 1, [3,6,7])
test(x,[[1,2,3],[1,7],[1,7]], 27)


x = tri_traversal(cost2, heuristic2, 1, [4,6])
test(x,[[1,2,3,4],[1,7,3,6],[1,7,3,4]], 28) # i donno a* can be = [1,7,3,6] cuz it costs the same f(n)


x = tri_traversal(cost2, heuristic2, 1, [2,3,7])
test(x,[[1,2],[1,7],[1,7]], 29)


# x = tri_traversal(cost2, heuristic2, 4, [3])
# test(x,[[],[],[]], 30)


x = tri_traversal(cost3, heuristic3, 1, [5])
test(x,[[1,2,3,4,5],[1,2,3,5],[1,2,3,5]], 31)


try:
    print("-------------------------------------------------------")
    file = open("mi_test_cases.txt", "r")
    test_num = 1
    failed = []
    file.readline()
    file.readline()
    file.readline()
    for i in range(10):
        file.readline()
        size = int(file.readline())
        file.readline()
        cost = [list(map(int, file.readline().split())) for x in range(size)]
        for j in range(10):
            file.readline()
            file.readline()
            file.readline()
            heuristic = list(map(int, file.readline().split()))
            file.readline()
            start_point = int(file.readline())
            file.readline()
            goals = list(map(int, file.readline().split()))
            file.readline()
            correct_answer = [list(map(int, s.split())) for s in file.readline().split(",")]
            correct_answer.pop()
            your_answer = tri_traversal(cost, heuristic, start_point, goals)
            if(your_answer == correct_answer):
                print("Test",test_num,": PASSED!")
            else:
                print("Test",test_num,": FAILED!")
                print("Your answer :", your_answer)
                print("Correct answer :", correct_answer)
                failed.append(test_num)
            test_num += 1
        file.readline()
        
    if(not failed):
        print("ALL TEST CASES PASSED!")
    else:
        print("ono FOLLOWING TEST CASES FAILED!")
        print(*failed)
except:
    print("probably an error in the code :( or Test cases may included goals unreachable from start point, in which case the result for each search is an empty list")
