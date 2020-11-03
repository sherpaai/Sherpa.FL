import random
import numpy as np
import tensorflow as tf
import torch


class Reproducibility:
    """
    Singleton class for ensure reproducibility.
    You indicates the seed and the execution is the same. The server initialice this class and the clients only
    call/get a seed.

    Server initialize it with Reproducibility(seed) before all executions
    For get a seed, the client has to put Reproducibility.get_instance().set_seed(ID)

    Is important to know that the reproducibility only works if you execute the experiment in CPU. Many ops in GPU
    like convolutions are not deterministic and the don't replicate.

    # Arguments:
        seed: the main seed for server

    # Properties:
        seed:
            return server seed
        seeds:
            return all seeds
    """
    __instance = None

    @staticmethod
    def get_instance():
        """
        Static access method.

        # Returns:
            instance: Singleton instance class
        """
        if Reproducibility.__instance is None:
            Reproducibility()
        return Reproducibility.__instance

    def __init__(self, seed=None):
        """
        Virtually private constructor.
        """
        if Reproducibility.__instance is not None:
            raise Exception("This class is a singleton")
        else:
            self.__seed = seed
            self.__seeds = {'server': self.__seed}
            Reproducibility.__instance = self

            if self.__seed is not None:
                self.set_seed('server')

    def set_seed(self, id):
        """
        Set server and clients seed

        # Arguments:
            id: 'server' in server node and ID in client node
        """
        if id not in self.__seeds.keys():
            self.__seeds[id] = np.random.randint(2**32-1)
        np.random.seed(self.__seeds[id])
        random.seed(self.__seeds[id])
        tf.random.set_seed(self.__seeds[id])
        torch.manual_seed(self.__seeds[id])

    @property
    def seed(self):
        return self.__seed

    @property
    def seeds(self):
        return self.__seeds

    def delete_instance(self):
        """
        Remove the singleton instance. Not recommended for normal use. This method is necessary for tests.
        """
        if Reproducibility.__instance is not None:
            del self.__seed
            del self.__seeds
            Reproducibility.__instance = None


