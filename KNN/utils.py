import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    #assert len(real_labels) == len(predicted_labels)
    #raise NotImplementedError
    real_labels = np.array(real_labels)
    predicted_labels = np.array(predicted_labels)
    tp = np.sum((predicted_labels == 1) & (real_labels == 1))
    tn = np.sum((predicted_labels == 0) & (real_labels == 0))
    fp = np.sum((predicted_labels == 1) & (real_labels == 0))
    fn = np.sum((predicted_labels == 0) & (real_labels == 1))
    recall = (tp/(tp + fn)) if tp != 0 else 0
    precision = (tp/(tp + fp)) if tp != 0 else 0
    
    if precision == 0 or recall == 0:
        return 0.0
    
    return 2 * (recall * precision)/(precision + recall)


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        dist = np.sum(np.abs(np.array(point1) - np.array(point2)) ** 3) ** (1/3)
        return dist

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        dist = np.linalg.norm(np.array(point1) - np.array(point2))
        return dist


    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        if np.linalg.norm(p1) == 0 or np.linalg.norm(p2) == 0:
            dist = 1.0
        else:
            dist = 1 - np.dot(p1, p2)/np.linalg.norm(p1)/np.linalg.norm(p2)                          

        return dist

    
class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        #raise NotImplementedError
        score_max = -1        
        for fn in distance_funcs:
            for k in range(1, 30, 2):
                model = KNN(k, distance_funcs[fn])
                model.train(x_train, y_train)
                predicted_labels = model.predict(x_val)
                score = f1_score(y_val, predicted_labels)
                if score > score_max:
                    score_max = score
                    self.best_k = k
                    self.best_distance_function = fn
                    self.best_model = model
                    
                """
                elif score == score_max:
                    dist_keys = list(distance_funcs.keys())                    
                    if (dist_keys.index(self.best_distance_function) > dist_keys.index(fn)) or (self.best_k > k):
                        self.best_k = k
                        self.best_distance_function = fn
                        self.best_model = model
               """

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        #raise NotImplementedError
        score_max = -1
        for scaler in scaling_classes:            
            for fn in distance_funcs:                
                for k in range(1, 30, 2):
                    if k > len(x_train):
                        break
                    model = KNN(k, distance_funcs[fn])
                    sc = scaling_classes[scaler]()
                    model.train(sc(x_train), y_train)
                    sc = scaling_classes[scaler]()
                    predicted_labels = model.predict(sc(x_val))
                    score = f1_score(y_val, predicted_labels)
                    if score > score_max:
                        score_max = score
                        self.best_k = k
                        self.best_distance_function = fn
                        self.best_model = model
                        self.best_scaler = scaler
                    """
                    elif score == score_max:                        
                        dist_keys = list(distance_funcs.keys())
                        if (self.best_scaler > scaler) or (dist_keys.index(self.best_distance_function) > dist_keys.index(fn)) or (self.best_k > k):
                            self.best_k = k
                            self.best_distance_function = fn
                            self.best_model = model
                            self.best_scaler = scaler"""

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError
        features_np = np.array(features)
        norm = np.linalg.norm(features_np, axis=1, keepdims=True)        
        norm[norm==0] = 1
        
        return (features_np/norm).tolist()


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError
        features_np = np.array(features)
        mini = features_np.min(axis=0)
        maxi = features_np.max(axis=0)
        diff = maxi - mini
        diff[diff == 0] = 1
        features_np[:, mini == maxi] == 0
(mini != maxi)])        
        features_np = (features_np - mini)/diff.reshape(1, diff.shape[0])
        return features_np.tolist()
