"""
• Create parameter server -> done
• Create replay buffer with shared memory
• Create a learning and sampler -> done
• Create main starting multiple processes
• Create
• Create
"""

from multiprocessing import Manager

if __name__ == '__main__':
    m = Manager()
    l = m.list()
    for i in range(10):
        l.append(i)

    print()