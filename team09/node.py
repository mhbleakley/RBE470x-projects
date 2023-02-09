class Node:
	
	def __init__(self, value):
		
		self.value = value
		self.dx = 0
		self.dy = 0
		self.mx = 0
		self.my = 0
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
	
# Driver code
# if __name__=='__main__':	
# 	root = Node(0, 0, 0, 0, 0)
# 	make_min(root, 5)
# 	print(root)
# 	res = expectimax(root, True)
# print("Expectimax value is "+str(res))
