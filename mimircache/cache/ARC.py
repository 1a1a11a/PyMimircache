import sys
import os

from collections import defaultdict
from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList


class ARC(cache):
    def __init__(self, cache_size=1000, p=0.5, ghostlist_size=-1):
        '''

        :param cache_size:
        :param p: the position to separate cache for LRU1 and LRU2
        :param ghostlist_size:
        :return:
        '''
        super().__init__(cache_size)
        self.p = p
        if ghostlist_size == -1:
            self.ghostlist_size = self.cache_size

        # dict/list2 is for referenced more than once
        self.linkedList1 = LinkedList()
        self.linkedList2 = LinkedList()
        self.lru_list_head_p1 = None
        self.lru_list_head_p2 = None

        self.cacheDict1 = dict()  # key -> linked list node (in reality, it should also contains value)
        self.cacheDict2 = dict()  # key -> linked list node (in reality, it should also contains value)
        self.cacheDict1_ghost = defaultdict(
            lambda: 0)  # key -> linked list node (in reality, it should also contains value)
        self.cacheDict2_ghost = defaultdict(
            lambda: 0)  # key -> linked list node (in reality, it should also contains value)

    def checkElement(self, element):
        '''
        :param element: the element for search
        :return: whether the given element is in the cache
        '''
        if element in self.cacheDict1 or element in self.cacheDict2:
            return True
        else:
            return False

    def check_ghost_list(self, element):
        '''
        :param element: the element for search
        :return: whether the given element is in the cache
        '''
        if element in self.cacheDict1_ghost or element in self.cacheDict2_ghost:
            return True
        else:
            return False

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: None
        '''
        if element in self.cacheDict1:
            # move to part2
            # get the node and remove from part1, size of part1 reduced by 1
            node = self.cacheDict1[element]
            # print("before remove")
            # for i in self.linkedList1:
            #     print(i.content, end='\t')
            # print("")
            if node == self.lru_list_head_p1:
                self.lru_list_head_p1 = self.lru_list_head_p1.next
            self.linkedList1.removeNode(node)
            # print("after remove")
            # for i in self.linkedList1:
            #     print(i.content, end='\t')
            # print("")

            del self.cacheDict1[element]

            # insert into part2
            self.linkedList2.insertNodeAtTail(node)
            self.cacheDict2[node.content] = node
            # delete one from part1, insert one into part2, the total size should not change, just the balance changes
            # check whether lru_list_head_p2 has been initialized or not
            if not self.lru_list_head_p2:
                self.lru_list_head_p2 = node


        else:
            node = self.cacheDict2[element]
            if node == self.lru_list_head_p2 and node.next:
                self.lru_list_head_p2 = self.lru_list_head_p2.next

            self.linkedList2.moveNodeToTail(node)

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: evicted element or None
        '''
        return_content = None
        if self.linkedList1.size + self.linkedList2.size >= self.cache_size:
            # needs to evict one element, depend on ghost list to decide evict from part1 or part2
            return_content = self._evictOneElement(element)

        # insert into part 1
        node = self.linkedList1.insertAtTail(element)
        self.cacheDict1[element] = node
        if not self.lru_list_head_p1:
            self.lru_list_head_p1 = node
        return return_content

    def printCacheLine(self):
        print('list 1(including ghost list): ')
        for i in self.linkedList1:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)
        print('\nlist 2(including ghost list): ')
        for i in self.linkedList2:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)
        print(' ')

    def _evictOneElement(self, element):
        '''
        evict one element from the cache line into ghost list, then check ghost list,
        if oversize, then evict one from ghost list
        :param: element: the missed request
        :return: content of element evicted into ghost list
        '''
        return_content = None
        if element in self.cacheDict1_ghost and len(self.cacheDict2) > 0:
            # evict one from part2 LRU, add into ghost list
            return_content = content = self.lru_list_head_p2.content
            del self.cacheDict2[content]
            self.lru_list_head_p2 = self.lru_list_head_p2.next
            self.cacheDict2_ghost[content] += 1

            if (self.linkedList2.size - len(self.cacheDict2)) > self.ghostlist_size:
                # the first part is the size of ghost list of part2
                content = self.linkedList2.removeFromHead()
                if self.cacheDict2_ghost[content] == 1:
                    del self.cacheDict2_ghost[content]
                else:
                    self.cacheDict2_ghost[content] -= 1
                assert (self.linkedList2.size - len(self.cacheDict2)) == self.ghostlist_size


        elif element in self.cacheDict2_ghost and len(self.cacheDict1) > 0:
            # evict one from part1 LRU, add into ghost list
            return_content = content = self.lru_list_head_p1.content
            del self.cacheDict1[content]

            self.lru_list_head_p1 = self.lru_list_head_p1.next
            self.cacheDict1_ghost[content] += 1

            if (self.linkedList1.size - len(self.cacheDict1)) > self.ghostlist_size:
                # the first part is the size of ghost list of part1
                content = self.linkedList1.removeFromHead()
                if self.cacheDict1_ghost[content] == 1:
                    del self.cacheDict1_ghost[content]
                else:
                    self.cacheDict1_ghost[content] -= 1
                assert (self.linkedList1.size - len(self.cacheDict1)) == self.ghostlist_size

        else:
            # not in any ghost list, check p value
            if len(self.cacheDict2) == 0 or len(self.cacheDict1) / len(self.cacheDict2) > self.p:
                # remove from part1
                return_content = content = self.lru_list_head_p1.content
                del self.cacheDict1[content]

                self.lru_list_head_p1 = self.lru_list_head_p1.next
                self.cacheDict1_ghost[content] += 1

                if (self.linkedList1.size - len(self.cacheDict1)) > self.ghostlist_size:
                    # the first part is the size of ghost list of part1
                    content = self.linkedList1.removeFromHead()
                    if self.cacheDict1_ghost[content] == 1:
                        del self.cacheDict1_ghost[content]
                    else:
                        self.cacheDict1_ghost[content] -= 1
                    assert (self.linkedList1.size - len(self.cacheDict1)) == self.ghostlist_size
            else:
                # remove from part2
                return_content = content = self.lru_list_head_p2.content
                del self.cacheDict2[content]

                self.lru_list_head_p2 = self.lru_list_head_p2.next
                self.cacheDict2_ghost[content] += 1

                if (self.linkedList2.size - len(self.cacheDict2)) > self.ghostlist_size:
                    # the first part is the size of ghost list of part2
                    content = self.linkedList2.removeFromHead()
                    if self.cacheDict2_ghost[content] == 1:
                        del self.cacheDict2_ghost[content]
                    else:
                        self.cacheDict2_ghost[content] -= 1
                    assert (self.linkedList2.size - len(self.cacheDict2)) == self.ghostlist_size

        return return_content

    # for debug
    def _check_lru_list_p(self):
        size1 = sum(i for i in self.cacheDict1_ghost.values())
        node1 = self.linkedList1.head.next
        for i in range(size1):
            node1 = node1.next
        assert node1 == self.lru_list_head_p1, "LRU list1 head pointer wrong"

        size2 = sum(i for i in self.cacheDict2_ghost.values())
        node2 = self.linkedList2.head.next
        for i in range(size2):
            node2 = node2.next
        print(size2)
        if node2:
            print(node2.content)
        else:
            print(node2)
        if self.lru_list_head_p2:
            print(self.lru_list_head_p2.content)
        else:
            print(self.lru_list_head_p2)
        assert node2 == self.lru_list_head_p2, "LRU list2 head pointer wrong"

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: None
        '''
        if self.checkElement(element):
            self._updateElement(element)
            # self.printCacheLine()
            # self._check_lru_list_p()
            return True
        else:
            self._insertElement(element)
            # self.printCacheLine()
            # self._check_lru_list_p()
            return False

    def __repr__(self):
        return "ARC, given size: {}, current part1 size: {}, part2 size: {}".format(
            self.cache_size, self.linkedList1.size, self.linkedList2.size)
