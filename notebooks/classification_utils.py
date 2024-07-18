
import numpy as np
import typing as th
from sklearn.ensemble import IsolationForest

def get_naive_accuracy(
    pos_x : th.Optional[np.ndarray] = None,
    pos_y : th.Optional[np.ndarray] = None,
    neg_x : th.Optional[np.ndarray] = None,
    neg_y : th.Optional[np.ndarray] = None,
    percentile: int = 10,
):
    """
    Create a dual threshold classifier by just looking at the training data
    """
    if pos_y is None or neg_y is None:
        pos_y = np.ones_like(pos_x)
        neg_y = np.ones_like(neg_x)
        
    x_threshold = np.sort(pos_x)[int(1.0 * percentile / 100 * len(pos_x))]
    y_threshold = np.sort(pos_y)[int(1.0 * percentile / 100 * len(pos_y))]
    
    tp = np.sum((pos_x >= x_threshold) & (pos_y >= y_threshold))
    fn = np.sum((pos_x < x_threshold) | (pos_y < y_threshold))
    tn = np.sum((neg_x < x_threshold) | (neg_y < y_threshold))
    fp = np.sum((neg_x >= x_threshold) & (neg_y >= y_threshold))
    
    return 1.0 * (tp + tn) / (tp + tn + fp + fn)

def get_isolation_forest_accuracy(
    pos_x : th.Optional[np.ndarray] = None,
    pos_y : th.Optional[np.ndarray] = None,
    neg_x : th.Optional[np.ndarray] = None,
    neg_y : th.Optional[np.ndarray] = None,
    contamination: float = 0.1,
):
    if pos_y is None or neg_y is None:
        pos_y = np.ones_like(pos_x)
        neg_y = np.ones_like(neg_x)
    
    # 1. Combine pos_x and pos_y to form the training data
    training_data = np.column_stack((pos_x, pos_y))

    # 2. Fit Isolation Forest classifier
    clf = IsolationForest(contamination=contamination) # you can adjust contamination based on expected outlier ratio
    clf.fit(training_data)
    
    # 3. Combine neg_x and neg_y to form the test data
    neg_data = np.column_stack((neg_x, neg_y))
    # 4. Predict anomalies in the test data
    # Note: In the result, -1 indicates an anomaly and 1 indicates a normal data point.
    neg_predictions = clf.predict(neg_data)
    pos_predictions = clf.predict(training_data)
    
    tp = np.sum(pos_predictions == 1)
    fn = np.sum(pos_predictions == -1)
    tn = np.sum(neg_predictions == -1)
    fp = np.sum(neg_predictions == 1)
     
    return 1.0 * (tp + tn) / (tp + tn + fp + fn)