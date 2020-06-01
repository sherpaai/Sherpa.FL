import random
import numpy as np
import tensorflow as tf


class Reproducibility:
    """
    Singleton class for ensure reproducibility.
    You indicates the seed and the execution is the same. The server initialice this class and the clients only
    call/get a seed.

    Server initialize it with Reproducibility(seed) before all executions
    For get a seed, the client has to put Reproducibility.getInstance().set_seed(ID)

    Is important to know that the reproducibility only works if you execute the experiment in CPU. Many ops in GPU
    like convolutions are not deterministic and the don't replicate.

    # Methods:
        getInstance():
            Return the singleton instance
        set_seed():
            The clients call this method for set a seed. They have to know their own ID's

    # Attributes:
        seed: Seed for the execution
        __instance: Singleton instance

    # Properties:
        seed:
            return server seed
        seeds:
            return all seeds
    """
    __instance = None

    @staticmethod
    def getInstance():
        """
        Static access method.

        Return:
            Singleton instance class
        """
        if Reproducibility.__instance == None:
            Reproducibility()
        return Reproducibility.__instance

    def __init__(self, seed=None):
        """
        Virtually private constructor.
        """
        if Reproducibility.__instance != None:
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

        Attributes:
            id: 'server' in server node and ID in client node
        """
        if id not in self.__seeds.keys():
            self.__seeds[id] = np.random.randint(2**32-1)
        np.random.seed(self.__seeds[id])
        random.seed(self.__seeds[id])
        tf.random.set_seed(self.__seeds[id])


    @property
    def seed(self):
        return self.__seed

    @property
    def seeds(self):
        return self.__seeds

    def delete_instance(self):
        if Reproducibility.__instance != None:
            del self.__seed
            del self.__seeds
            Reproducibility.__instance = None


