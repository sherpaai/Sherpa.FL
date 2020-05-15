from shfl.learning_approach.federated_images_classifier import FederatedImagesClassifier, ImagesDataBases
import random
import string


def test_images_classifier():
    # Test initialiser
    example_database = list(ImagesDataBases.__members__.keys())[0]

    assert FederatedImagesClassifier(example_database)
    assert FederatedImagesClassifier(example_database, False)

    letters = string.ascii_lowercase
    wrong_database = ''.join(random.choice(letters) for i in range(10))

    assert FederatedImagesClassifier(wrong_database)

    # Test run rounds
    classifier = FederatedImagesClassifier(example_database)
    assert classifier.run_rounds(1)
