
# Question 1
# Given two strings s and t, determine whether some anagram of t is a substring of s. 
# For example: if s = "udacity" and t = "ad", then the function returns True. 
# Your function definition should look like: question1(s, t) and return a boolean True or False.

def is_anagram(s, t):
    
    ''' Determine if string s and t are anagram, by compare count of each characters '''

    import string
    d_s = {}
    d_t = {}
    
    alp = string.ascii_lowercase
    for c in alp:
        d_s[c] = 0
        d_t[c] = 0
    
    for c in t:
        d_t[c]+=1
    
    for c in s:
        d_s[c]+=1
    
    for c in alp:
        if d_s[c] != d_t[c]:
            return False
    
    return True


def question1(s, t):
    
    ''' Loop through all consecutive substring of s and if it's anagram of t, return True '''

    len_s, len_t = len(s), len(t)
    
    if len_s < len_t or len_t == 0:
        return False
    
    for i in range(len_s - len_t +1):
        # print s[i:i+len_t], t
        if is_anagram(s[i:i+len_t],t):
            return True
    
    return False

print "Question 1: "
print "Test1: regular case, s = 'udacity', t = 'ad'"
print question1("udacity", "ad")
# True
print "Test2: Empty t string, s = 'udacity', t = ''"
print question1("udacity", "")
# False

print "Test3: re-arranged original list, s = 'udacity', t = 'dautyci'"
print question1("udacity", "dautyci")
# True

print "Test4: long t, s = 'udacity', t = 'dautycihahaha'"
print question1("udacity", "dautycihahaha")
# False
print "--------------------------------------"

# Question 2
# Given a string a, find the longest palindromic substring contained in a. 
# Your function definition should look like question2(a), and return a string.


def is_palindromic(string):
    ''' Palindromic string will keep same with reversed order '''
    return string[::] == string[::-1]

def question2(a):

    ''' Palindromic string will keep same with reversed order '''
    a = a.lower()
    len_a = len(a)
    
    for l in reversed(range(len_a + 1)):
        for i in range(len_a - l + 1):
            if is_palindromic(a[i:i+l]):
                return a[i:i+l]

print "Question 2: "
print "Test1: itself, a = 'aba'"
print question2("aba")
# aba
print "Test2: empty input, a = ''"
print question2("")
# (nothing)
print "Test3: long input, a = 'refea sffsa fvasabacabasdr'"
print question2("refea sffsa fvasabacabasdr")
# sabacabas
print "Test3: short input, a = 'ab'"
print question2("ab")
# a
print "--------------------------------------"


# Question 3
# Given an undirected graph G, find the minimum spanning tree within G. 
# A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. 
# Your function should take in and return an adjacency list structured like this:

# {'A': [('B', 2)],
#  'B': [('A', 2), ('C', 5)], 
#  'C': [('B', 5)]}
# Vertices are represented as unique strings. The function definition should be question3(G)

# Kruskal's algorithm
## 1. Input: Graph Object
class Node(object):
    def __init__(self, value):
        self.value = value
        self.edges = []

class Edge(object):
    def __init__(self, value, node_from, node_to):
        self.value = value
        self.node_from = node_from
        self.node_to = node_to

class Graph(object):
    def __init__(self):
        self.nodes = []
        self.edges = []

    def insert_node(self, new_node_val):
        new_node = Node(new_node_val)
        self.nodes.append(new_node)
        
    def insert_edge(self, new_edge_val, node_from_val, node_to_val):
        from_found = None
        to_found = None
        for node in self.nodes:
            if node_from_val == node.value:
                from_found = node
            if node_to_val == node.value:
                to_found = node
        if from_found == None:
            from_found = Node(node_from_val)
            self.nodes.append(from_found)
        if to_found == None:
            to_found = Node(node_to_val)
            self.nodes.append(to_found)
        new_edge = Edge(new_edge_val, from_found, to_found)
        from_found.edges.append(new_edge)
        to_found.edges.append(new_edge)
        self.edges.append(new_edge)

    def get_edge_list(self):
        return [(edge.value, edge.node_from.value, edge.node_to.value) for edge in self.edges]
    
    def get_node_list(self):
        return [node.value for node in self.nodes]

# use UnionFind algorithm to detect if a new added edge will create a circle with existed edges
def make_circuit(existed_edges, edge):
        
    # 2. Union Find to judge if there's a circuit created by new edge and existed edges

    parent = {}
    edges = existed_edges + [edge]
    for node_value in nodes_of(edges):
        parent[node_value] = -1
    
    def find_parent(node_value):
        if parent[node_value] == -1:
            return node_value
        else:
            return find_parent(parent[node_value])
    
    def union(x, y):
        x_set = find_parent(x)
        y_set = find_parent(y)
        parent[x_set] = y_set

        
    for edge in edges:
        x = find_parent(edge[1])
        y = find_parent(edge[2])
        
        if x == y:
            return True
        else:
            union(x, y)
    
    return False

def nodes_of(edge_lst):
        node_lst = []
        for edge in edge_lst:
            node_lst.append(edge[1])
            node_lst.append(edge[2])
        return sorted(list(set(node_lst)))

def question3(G):
        
    # 1. get sorted list of edges, element will be (weight, node_to, node_from)
    sorted_edge_lst = sorted(G.get_edge_list())


    # 2. iterate to pickup one edge from sorted edge list, until all nodes reached
    ##      if this new edge doesn't makes a circuit with existed edges, add to existed edges
    ##      else, do nothing
    ##      if all nodes got reached, break

    # initialize an empty list as existed edges to append edges from sorted edge list
    existed_edges = []

    # used to extract nodes and compare if all nodes are reached
    nodes_lst = G.get_node_list()

    for edge in sorted_edge_lst:
        if not make_circuit(existed_edges, edge):
            existed_edges.append(edge)

        if nodes_of(existed_edges) == nodes_lst:
            break;

    # 3. format output (return), from edge list to the required format
    # {'A': [('B', 2)], 'B': [('A', 2), ('C', 5)], 'C': [('B', 5)]}
    
    nodesValues = nodes_of(existed_edges)
    adict = {}
    
    for node in nodesValues:
        adict[node] = None

    for edge in existed_edges:
        if adict[edge[1]]:
            adict[edge[1]].append((edge[2], edge[0]))
        else:
            adict[edge[1]] = [(edge[2], edge[0])]
            
        if adict[edge[2]]:
            adict[edge[2]].append((edge[1], edge[0]))
        else:
            adict[edge[2]] = [(edge[1], edge[0])]

    return adict

print "Question 3: "
# Test Case 1: normal case with 1 connected graph

graph1 = Graph()
graph1.insert_edge(100, 'A', 'B')
graph1.insert_edge(101, 'A', 'C')
graph1.insert_edge(102, 'A', 'D')
graph1.insert_edge(103, 'C', 'D')
print question3(graph1)
# {'A': [('B', 100), ('C', 101), ('D', 102)],
# 'B': [('A', 100)],
# 'C': [('A', 101)],
# 'D': [('A', 102)]}

# Test Case 2: 1 edge
graph2 = Graph()
graph2.insert_edge(100, 'A', 'C')
print question3(graph2)

# Test Case 3: 0 edge
graph3 = Graph()
print question3(graph3)

# Test Case 4: 2 disjoint edges (not in same set), will grow tree in each set seperately
graph4 = Graph()
graph4.insert_edge(100, 'A', 'C')
graph4.insert_edge(101, 'B', 'D')
print question3(graph4)
print "--------------------------------"


# Question 4
# Find the least common ancestor between two nodes on a binary search tree. 
# The least common ancestor is the farthest node from the root that is an ancestor of both nodes. 
# For example, the root is a common ancestor of all nodes on the tree, 
# but if both nodes are descendents of the root's left child, then that left child might be the 
# lowest common ancestor. You can assume that both nodes are in the tree, 
# and the tree itself adheres to all BST properties. 
# The function definition should look like question4(T, r, n1, n2), 
# where T is the tree represented as a matrix, where the index of the list is equal to the integer 
# stored in that node and a 1 represents a child node, r is a non-negative integer representing the root, 
# and n1 and n2 are non-negative integers representing the two nodes in no particular order. 
# For example, one test case might be

# question4([[0, 1, 0, 0, 0],
#            [0, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0],
#            [1, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0]],
#           3,
#           1,
#           4)
# and the answer would be 3.

print "Question 4: "

def question4(T, r, n1, n2):
    ''' If root larger than maximum of n1 and n2, traverse left, 
        If root smaller than minimum of n1 and n2, traverse right
        Else, when root is between two nodes, it's the least common ancestor '''

    if not is_matrix(T):
        print "Invalid Matrix"
        return
    
    if not is_tree(T):
        print "Invalid Tree Matrix"
        return
        
    if not is_bst(T):
        print "Invalid Binary Search Tree"
        return
    
    # Step 1. find ancesstors of n1 and ancesstors of n2

    
    #      - make sure n1 and n2 has such number
    
    sum_n1 = sum([elem[n1] for elem in T])
    sum_n2 = sum([elem[n2] for elem in T])
    
    if sum_n1 < 1:

        print "No such n1"
        return
    
    if sum_n2 < 1:
        print "No such n2"
        return    
    
    answer = question4_helper(T, r, n1, n2)
            
    return answer
        
def is_tree(m):
    for j in range(len(m[0])):
        s = 0
        for i in range(len(m)):
            s += m[i][j]
        
        if s > 1:
            return False
        else:
            return True

def is_matrix(m):
    c = len(m[0])
    for i in range(len(m)):
        if len(m[i]) != c:
            return False
    return True   

def is_bst(m):
    
    # 1. sum of each row must be less than 2 to make a binary tree
    for i in range(len(m)):
        if sum(m[i]) > 2:
            return False
        
        elif sum(m[i]) == 2:
            row_index = []
            for j in range(len(m[0])):
                if m[i][j] == 1:
                    row_index.append(j)
                    
        
            
    # 2. each node value must be between its left and right value, if it has
    #    which means the row number must be between column ids of cells == 1
            if min(row_index) >= i or max(row_index) <= i:
                return False
    
    return True


# 
def question4_helper(T, r, n1, n2):
    
    if r > max(n1, n2):
        # traverse left
        for j in range(len(T[0])):
            if T[r][j] == 1:
                r = j
                break;
        return question4_helper(T, r, n1, n2)
                
    elif r < min(n1, n2):
        # traverse right
        for j in reversed(range(len(T[0]))):
            if T[r][j] == 1:
                r = j
                break;
        return question4_helper(T, r, n1, n2)
    else:
        return r
        


# test 1: regular
print "Test 1: regular"
T = [[0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]
r =3
n1 = 1
n2 = 4
print "T:"
for i in range(len(T)):
    print T[i]

print "r: %d" % r
print "n1: %d" % n1
print "n2: %d" % n2
print "Result: "
print question4(T, r, n1, n2)
print "\n\n"


# test 2: invalid matrix
print "Test 2: invalid matrix"
T = [[0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]
r =3
n1 = 1
n2 = 4
print "T:"
for i in range(len(T)):
    print T[i]

print "r: %d" % r
print "n1: %d" % n1
print "n2: %d" % n2
print "Result: "
print question4(T, r, n1, n2)
print "\n\n"


# test 3: invalid bst
print "Test 3: invalid bst"


T = [[0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]
r =3
n1 = 1
n2 = 4
print "T:"
for i in range(len(T)):
    print T[i]

print "r: %d" % r
print "n1: %d" % n1
print "n2: %d" % n2
print "Result: "
print question4(T, r, n1, n2)
print "\n\n"



# test 4: More complex matrix, should print 5
print "Test 4: More complex 12x12 matrix"

T = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

r = 11
n1 = 1
n2 = 8
print "T:"
for i in range(len(T)):
    print T[i]

print "r: %d" % r
print "n1: %d" % n1
print "n2: %d" % n2
print "Result: "
print question4(T, r, n1, n2)
print "\n\n"


# test 5: Reverse n1 and n2, should print 3
print "Test 5: Reverse n1 and n2, should print 5"

T = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

r = 11
n1 = 8
n2 = 1
print "T:"
for i in range(len(T)):
    print T[i]

print "r: %d" % r
print "n1: %d" % n1
print "n2: %d" % n2
print "Result: "
print question4(T, r, n1, n2)
print "\n\n"


# test 6: Empty matrix
print "Test 6: Empty matrix"
T = [[]]
r =3
n1 = 1
n2 = 4
print "T:"
for i in range(len(T)):
    print T[i]

print "r: %d" % r
print "n1: %d" % n1
print "n2: %d" % n2
print "Result: "
print question4(T, r, n1, n2)
print "\n\n"

question4([[]],
          6,
          4,
          0)
print "--------------------------------"

# Question 5
# Find the element in a singly linked list that's m elements from the end. 
# For example, if a linked list has 5 elements, the 3rd element from the end is the 3rd element. 
# The function definition should look like question5(ll, m), 
# where ll is the first node of a linked list and m is the "mth number from the end". 
# You should copy/paste the Node class below to use as a representation of a node in the linked list. 
# Return the value of the node at that position.

# class Node(object):
#   def __init__(self, data):
#     self.data = data
#     self.next = None


# idea: use stack
print "Question 5: "
class Node(object):
    def __init__(self, data):
        self.data = data
        self.next = None
        
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
        self.length = 1
        
    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
            
        else:
            self.head = new_element
        self.length +=1
            

def question5(ll, m):
    
    if m < 0:
        print "Wrong m value"
        return None
    
    if m > ll.length:
        return None
    
    current = ll.head
    for i in range(ll.length - m):
        current = current.next
    
    return current.data

# Test1: regular

e1 = Node(1)
e2 = Node(2)
e3 = Node(3)
e4 = Node(4)
ll = LinkedList(e1)
ll.append(e2)
ll.append(e3)
ll.append(e4)
print "Test1: regular, ll has value 1,2,3,4, so when m = 3, data is 2"
print question5(ll, 3)
print "\n"

# Test2: exceed length, print None if m is too large
print "Test2: exceed length, print None if m is too large"
e1 = Node(1)
e2 = Node(2)
e3 = Node(3)
e4 = Node(4)
ll = LinkedList(e1)
ll.append(e2)
ll.append(e3)
ll.append(e4)
# ll has value 1,2,3,4, so when m = 3, data is 2
print question5(ll, 5)
print "\n"


# Test3: Empty linked list, print None since there's no data
print "Test3: Empty linked list, print None since there's no data"
e1 = None
ll = LinkedList(e1)
print question5(ll, 3)
print "\n"







