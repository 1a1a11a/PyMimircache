class LinkedListNode:
    def __init__(self):
        self.content = None
        self.id = -1
        self.next = None
        self.prev = None




class LinkedList:
    def __init__(self, maxlen=-1):
        '''
        initialization of the linkedlist, head does not contain any element, but head contains last element
        :return:
        '''
        self.head = LinkedListNode()
        self.tail = None
        self.size = 0
        self.currentNode = self.head
        # self.max_size = maxlen


    def insertAtTail(self, content):
        node = LinkedListNode()
        node.content = content
        node.next = None
        node.prev = self.tail
        if self.tail:
            self.tail.next = node

        self.tail = node
        self.size += 1

        # if self.max_size!=-1 and self.size>self.max_size:
        #     self.removeFromTail()


    def insertAtHead(self, content):
        node = LinkedListNode()
        node.content = content
        node.next = self.head.next
        if self.head.next:
            self.head.next.prev = node
        else:
            self.tail = node
        self.head.next = node
        node.prev = self.head
        self.size += 1
        return node



    def removeFromTail(self):
        tail = self.tail
        self.tail.prev.next = None
        temp = self.tail.prev
        self.tail.prev = None
        self.tail = temp
        self.size -= 1
        return tail.content


    def removeFromHead(self):
        headContent = self.head.next.content
        temp = self.head.next.next
        self.head.next.prev = None
        self.head.next.next = None
        self.head.next = temp
        temp.prev = self.head
        self.size -= 1
        return headContent


    def moveNodeToHead(self, node):
        if self.head.next != node:
            node.prev.next = node.next
            if node.next:
                node.next.prev = node.prev
            else:
                self.tail = node.prev

            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            node.prev = self.head


    def getHead(self):
        return self.head.next.content

    def getTail(self):
        return self.tail.content



    def __iter__(self):
        self.currentNode = self.head
        return self



    def next(self):
        return self.__next__()




    def __next__(self):                     # Python 3
        if self.currentNode.next == None:
            # self.currentNode = self.head
            raise StopIteration
        else:
            self.currentNode = self.currentNode.next
            return self.currentNode