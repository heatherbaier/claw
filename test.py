# from multiprocessing import Process, Array, Pipe, Lock
import torch.multiprocessing as mp


def here(memory, val, lock, limit = 5):

    with lock:
        memory.append(val)

        if len(memory) == limit + 1:
            memory.pop(0)

    print(memory)


if __name__ == '__main__':

    with mp.Manager() as manager:
        

        memory = manager.list()
        processes = []
        lock = mp.Lock()

        for val in range(0, 10):
            p = mp.Process(target = here, args = (memory, val, lock, ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()