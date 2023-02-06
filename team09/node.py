class Node:
	
	def __init__(self, value):
		
		self.value = value
		self.N = None
		self.NE = None
		self.E = None
		self.SE = None
		self.S = None
		self.SW = None
		self.W = None
		self.NW = None
		self.B = None #Bomb
		self.D = None #Don't move
	
# Initializing Nodes to None
def newNode(v):

	temp = Node(v)
	return temp

# Getting expectimax
def expectimax(node, is_max):

	# Condition for Terminal node
	if (node.N == None and node.NE == None and node.E == None and node.SE == None and node.S == None and node.SW == None and node.W == None and node.NW == None and node.B == None and node.D == None):
		return node.value
	
	# Maximizer node. Chooses the max from the
	# left and right sub-trees
	if (is_max):
		return max(expectimax(node.N, False), expectimax(node.NE, False), expectimax(node.E, False), expectimax(node.SE, False), expectimax(node.S, False), expectimax(node.SW, False), expectimax(node.W, False), expectimax(node.NW, False), expectimax(node.B, False), expectimax(node.B, False))

	# Chance node. Returns the average of
	# the left and right sub-trees
	else:
		return (expectimax(node.N, True), expectimax(node.NE, True), expectimax(node.E, True), expectimax(node.SE, True), expectimax(node.S, True), expectimax(node.SW, True), expectimax(node.W, True), expectimax(node.NW, True), expectimax(node.B, True), expectimax(node.B, True))/10
	
# def make_min(nd):
# 	nd.N = newNode(0, 0, -1)
# 	nd.NE = newNode(0, 1, -1)
# 	nd.E = newNode(0, 1, 0)
# 	nd.SE = newNode(0, 1, 1)
# 	nd.S = newNode(0, 0, 1)
# 	nd.SW = newNode(0, -1, 1)
# 	nd.W = newNode(0, -1, 0)
# 	nd.NW = newNode(0, -1, -1)
# 	nd.B = newNode(0, 0, -1)
# 	nd.D = newNode(0, 0, -1)

def populate(node, height):
	if height == 0:
		node = newNode(0)
	else:
		node = newNode(height)
		for i in range(0,height):
			populate(node.N, height - 1)	
			populate(node.NE, height - 1)	
			populate(node.E, height - 1)	
			populate(node.SE, height - 1)	
			populate(node.S, height - 1)	
			populate(node.SW, height - 1)	
			populate(node.W, height - 1)	
			populate(node.NW, height - 1)	
			populate(node.B, height - 1)	
			populate(node.D, height - 1)
		
# Driver code
if __name__=='__main__':	
	root = newNode(0)
	populate(root, 3)
	print(root)
	res = expectimax(root, True)
print("Expectimax value is "+str(res))
