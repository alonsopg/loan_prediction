import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Perform classification with a gradient boosting classifier
gbc1 = GradientBoostingClassifier(learning_rate=0.46, max_features=0.94, min_weight_fraction_leaf=0.0, n_estimators=500, random_state=42)
gbc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['gbc1-classification'] = gbc1.predict(result1.drop('class', axis=1).values)

# Use Scikit-learn's StandardScaler to scale the features
training_features = result1.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    scaler = StandardScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform(result1.drop('class', axis=1).values.astype(np.float64))
    result2 = pd.DataFrame(data=scaled_features)
    result2['class'] = result1['class'].values
else:
    result2 = result1.copy()
