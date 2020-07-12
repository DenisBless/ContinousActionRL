"""
• Create parameter server -> done
• Create replay buffer with shared memory
• Create a learning and sampler -> done
• Create main starting multiple processes
• Create
• Create
"""

from multiprocessing import Manager, Lock
import multiprocessing as mp


def cube(a, u, lock):
    for z in a:
        with lock:
            if len(u) >= 20:
                return

            print(len(u))
            u.append(z ** 3)


def cube_tensor(a, t, lock, idx):
    for z in a:
        with lock:
            # if idx.item() > 5:
            #     return
            t[idx] = z**3
            idx += 1
            print(idx)


if __name__ == '__main__':

    import torch

    idx = torch.tensor(0)
    idx.share_memory_()

    t = torch.zeros([40, 1])
    t.share_memory_()
    lock = Lock()
    processes = [mp.Process(target=cube_tensor, args=(range(10), t, lock, idx))
                 for _ in range(4)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print(t)

# if __name__ == '__main__':
#
#     m = Manager()
#     # l = []
#     l = m.list()
#     lock=Lock()
#     processes = [mp.Process(target=cube, args=(range(10), l, lock))
#                  for _ in range(4)]
#
#     for p in processes:
#         p.start()
#
#     for p in processes:
#         p.join()
#
#     print(l)
#     print(l)
#     print(len(l))
