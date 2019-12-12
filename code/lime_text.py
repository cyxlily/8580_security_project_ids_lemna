import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
#from __future__ import print_function


from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
print(train_vectors.shape)#(1079, 23035)
print(test_vectors.shape)#(717, 23035)
#print(train_vectors[0])
#print(test_vectors[0])
print(newsgroups_train.target.shape)#(1079,)
print(newsgroups_train.target)#[1 0 1 ... 0 1 1]

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)

print(c.predict_proba([newsgroups_test.data[0]]))

from lime.lime_text import LimeTextExplainer
from lime.lime_text import IndexedString,TextDomainMapper,IndexedCharacters
from lime.lime_text import explanation
import scipy as sp

class LimeRnnExplainer(LimeTextExplainer):
    def explain_instance(self,text_instance,classifier_fn,labels=(1,),top_labels=None,num_features=25,num_samples=5000,distance_metric='cosine',model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw tabular data to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        indexed_string = (IndexedCharacters(text_instance, bow=self.bow, mask_string=self.mask_string) if self.char_level else IndexedString(text_instance, bow=self.bow,split_expression=self.split_expression,mask_string=self.mask_string))
        domain_mapper = TextDomainMapper(indexed_string)#change tabel data to string
        data, yss, distances = self.__data_labels_distances(indexed_string, classifier_fn, num_samples,distance_metric=distance_metric)
        print('data shape: ',data.shape)#(5000, 63)
        print('yss shape: ',yss.shape)#(5000,2)
        print('distances shape: ',distances.shape)#(5000,)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,class_names=self.class_names,random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],ret_exp.local_exp[label],ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(data, yss, distances, label, num_features,model_regressor=model_regressor,feature_selection=self.feature_selection)
        return ret_exp
    def __data_labels_distances(self,indexed_string,classifier_fn,num_samples,distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_words()
        print('doc_size',doc_size)
        sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)
        data = np.ones((num_samples, doc_size))#mask
        data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        inverse_data = [indexed_string.raw_string()]
        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(features_range, size,replace=False)
            print('inactive', inactive)#[ 8 62  5  3 37 39 16 44 58 48 32 57 50 18 38 34 40 61 43 15 49 59 13 12 51 27 35 36  6  2  7  4 56]
            data[i, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))
        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        print('data',data)
        print('labels',labels)
        print('distances',distances)
        return data, labels, distances

#explainer = LimeTextExplainer(class_names=class_names)
explainer = LimeRnnExplainer(class_names=class_names)

idx = 83
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)

print('Document id: %d' % idx)#83
print('Document: ',newsgroups_test.data[idx])#From: johnchad@triton.unm.edu (jchadwic)...
print(type(newsgroups_test.data[idx]))
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])#0.504
print('True class: %s' % class_names[newsgroups_test.target[idx]])#atheism
print(exp.as_list())
'''
print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])

exp.save_to_file('lime_text.html')
'''