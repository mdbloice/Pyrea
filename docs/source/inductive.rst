Inductive assignment
====================

``Graph.fit`` learns on training samples; ``Graph.predict`` assigns new samples
to frozen cluster identities. A single patient must have shape ``(1, p)`` in
each feature view. Prediction does not fit preprocessing, rerun clustering,
or perform genetic search. Standard fitted blocks operate on each new sample
independently of other samples in the prediction batch.

A fitted graph is an out-of-sample extension of a transductive ensemble. In
particular, hierarchical and spectral clustering have no native ``predict``.
The assignment rules below are explicit extensions, rather than reruns of those
algorithms with the new patient included.

Build, fit, and predict
-----------------------

.. code-block:: python

   import numpy as np
   from pyrea import nn

   class PatientEnsemble(nn.Module):
       def __init__(self):
           self.gene_scale = nn.Standardize()
           self.lipid_scale = nn.Standardize()
           self.gene_cluster = nn.Hierarchical(2, method='ward')
           self.lipid_cluster = nn.Hierarchical(2, method='complete')
           self.fusion = nn.Fusion('disagreement')
           self.first = nn.Hierarchical(2, input_kind=nn.DISTANCE)
           self.second = nn.Hierarchical(2, 'complete', input_kind=nn.DISTANCE)
           self.consensus = nn.Consensus()

       def forward(self, gene, lipid):
           first = self.gene_cluster(self.gene_scale(gene))
           second = self.lipid_cluster(self.lipid_scale(lipid))
           distance = self.fusion(first, second)
           labels = self.consensus(self.first(distance), self.second(distance))
           return {'labels': labels, 'distance': distance}

   training = np.array([[-4., 0.], [-3., 1.], [-2., 0.],
                        [2., 0.], [3., 1.], [4., 0.]])
   model = PatientEnsemble().compile()
   model.fit(gene=training, lipid=training + 100, assignment='nearest')
   patient = {'gene': np.array([[3.2, 0.1]]),
              'lipid': np.array([[103.2, 100.1]])}
   assigned = model.predict(patient)
   assert assigned.shape == (1,)
   assert np.isin(assigned, model.labels_).all()
   result = model.transform(patient)
   assert result['distance'].shape == (1, len(training))

``fit`` returns the graph itself and accepts an input mapping or named views.
It fits a copied snapshot, leaving the configured blocks available for ordinary
transductive calls. ``labels_`` and ``training_output_`` return copies of the
training result; ``n_samples_fit_`` records the training sample count.
``fit_predict`` returns the selected training labels without reassigning them.

``predict`` selects the named ``labels`` output, the graph's single label output,
or its single label result. If several named label outputs exist without a
``labels`` key, specify ``output='name'``. ``transform`` returns all declared
outputs; use ``return_intermediates=True`` to inspect the fitted pipeline.
Prediction before fitting raises ``NotFittedError``.

``run`` and ordinary graph calls remain transductive, even after fitting. Use
``predict`` or ``transform`` for frozen inference. A later ``fit`` replaces the
snapshot; if that fit fails, prediction requires a successful fit again.
``with_params`` returns an unfitted configuration copy and does not carry old
training artifacts into a new configuration. The fitted snapshot captures the
configuration at fit time; changing configured blocks requires fitting again.

Frozen clustering assignments
------------------------------

For built-in ``Cluster`` and ``Hierarchical`` blocks, ``assignment`` can be:

* ``'nearest'``: transfer the label of the nearest training sample.
* ``'average'``: choose the training cluster with the smallest mean distance
  from the new sample to its members.
* ``'medoid'``: select one training representative per cluster minimizing its
  total within-cluster distance, then assign to the nearest representative.

For precomputed affinity inputs, replace minimum distance with maximum affinity;
average uses mean affinity and the representative maximizes within-cluster
affinity. For feature inputs, hierarchical blocks use ``distance_metric``;
DBSCAN and OPTICS use their configured ``metric``, ``metric_params``, and
applicable Minkowski ``p``. Spectral feature inputs use Euclidean label transfer,
independently of the spectral training kernel. This is not a Nyström extension.
Variance/covariance needed for standardized Euclidean or Mahalanobis assignment
is frozen from training, unless already supplied in the metric parameters.

An override mapping uses names from ``named_blocks()``; unspecified blocks use
nearest assignment:

.. code-block:: python

   model.fit(
       gene=training, lipid=training + 100,
       assignment={'gene_cluster': 'medoid', 'first': 'average'},
   )
   assert model.predict(patient).shape == (1,)

Ties use training order: the first training sample, first observed cluster,
or first within-cluster representative as applicable. No new cluster is created.
These rules do not perform novelty detection. For DBSCAN and OPTICS they transfer
the frozen labels, including any training noise label; they do not reproduce
DBSCAN's core-radius membership rule or an OPTICS reachability extension.

``fit_predict`` records the transductive training partition. Average or medoid
assignment can reassign some training points if they are passed through
``predict``; label transfer is a separate operation from learning that partition.
At an individual block, nearest assignment recovers the label of a unique
zero-distance training match. Duplicate or zero-distance samples with different
training labels can tie.

Retained preprocessing and native prediction
--------------------------------------------

``nn.Transform(transformer)`` copies and fits a sklearn-style transformer during
training, then calls only ``transform`` during prediction. ``nn.Standardize``
wraps ``StandardScaler``. Means and scales are learned from training data, so
one new sample is not centered using its own values.

``nn.Estimator(estimator)`` adapts a clustering estimator with native ``fit``
and ``predict``, such as KMeans. Training uses ``fit_predict`` if available;
otherwise it fits and uses ``labels_`` or native prediction for training labels.
Prediction calls only the retained estimator's native ``predict``. Estimators
without native prediction should use the explicit label-transfer blocks instead.

.. code-block:: python

   from sklearn.cluster import KMeans
   from sklearn.preprocessing import StandardScaler

   x = nn.Input('data')
   features = nn.Transform(StandardScaler(), name='scale')(x)
   labels = nn.Estimator(
       KMeans(n_clusters=2, n_init=10, random_state=42), name='kmeans'
   )(features)
   native = nn.Graph({'data': x}, labels).fit(data=training)
   assert native.predict(data=np.array([[3., 0.]])).shape == (1,)

Each executable call has its own fitted state. Two calls reusing the same block
share configuration but fit independently on their respective training inputs.
A shared symbolic result is still computed once per invocation.

Fusion and consensus at prediction time
---------------------------------------

With ``m`` new and ``n`` training samples, pairwise prediction values have shape
``(m, n)``. Disagreement compares each new branch label with each training label:

.. math::

   D^{\mathrm{new,train}}_{ij}
     = \sum_h \mathbf{1}\{y^{\mathrm{new},h}_i \ne y^{\mathrm{train},h}_j\}.

Agreement counts matches, and consensus fusion requires a match in every
partition. Normalization uses the configured number of partitions. No positional
diagonal is zeroed: row zero is the first new patient, not training patient zero.
Even when ``m == n``, this is a cross-pairwise matrix rather than a within-batch
matrix. At training time, outputs retain the original square-matrix semantics.

``AffinityToDistance`` retains its training maximum, applies
``max(training_maximum - new_affinity, 0)``, and uses the frozen maximum for
optional normalization. Affinities exceeding the fitted maximum saturate at
zero distance. It neither estimates a new batch maximum nor overwrites a diagonal.

Final ``Consensus`` learns a membership-tuple-to-label mapping. Known tuples
receive their exact learned label. An unseen combination is assigned to the
observed training tuple with the fewest disagreeing components; ties use first
training occurrence. Set ``consensus='raise'`` in ``fit`` to reject unseen
combinations instead. This fallback is an explicit assignment approximation;
it does not expand the learned set of consensus clusters.

Schemas, persistence, and custom blocks
---------------------------------------

Feature inputs retain training feature counts and, for DataFrames, column names
and order. Identified query views must have matching sample IDs/order. A
precomputed query input must be new-by-training; identified columns must follow
the learned training sample order. Bare arrays are positional, so users remain
responsible for their column meanings. All required views must be provided.

Training references are copied, and prediction uses its fitted snapshot.
Training data, preprocessing state, label sets, and consensus mappings remain
unchanged by standard prediction operations. Fitted models can be serialized
with Python pickle when their components are picklable:

.. code-block:: python

   import pickle

   restored = pickle.loads(pickle.dumps(model))
   assert np.array_equal(restored.predict(patient), model.predict(patient))

Custom blocks implement ``fit_forward`` and ``predict_forward``. Training returns
the training output while retaining the state needed for future queries.
Prediction uses only that state and each sample's inputs. It must preserve sample
order, avoid fitting or batch-derived statistics, and return labels from the
learned set if its output kind is ``LABELS``. ``predict_forward`` for pairwise
outputs must return a new-by-training matrix. Prediction validates the shape and
label vocabulary. Mutable callable closures are not isolated automatically.

.. code-block:: python

   class Center(nn.Block):
       input_kinds = {'data': nn.FEATURES}
       output_kind = nn.FEATURES

       def forward(self, data):
           return data - data.mean(axis=0)

       def fit_forward(self, data):
           self.mean_ = data.mean(axis=0)
           return data - self.mean_

       def predict_forward(self, data):
           return data - self.mean_

   source = nn.Input('data')
   custom = nn.Graph({'data': source}, Center()(source)).fit(data=training)
   assert np.allclose(custom.transform(data=training[:1]),
                      training[:1] - training.mean(axis=0))

An arbitrary ``Function`` is not automatically inductive. Set ``inductive=True``
only for a stateless, sample-wise function that obeys the same contract. A function
that fits KMeans or calculates query-batch means does not satisfy it. Unsupported
blocks raise an error during fitting.

Genetic search and evaluation
-----------------------------

The existing genetic search evaluates transductive candidate graphs. Its returned
``best_graph`` is unfitted; fit it on the training data before prediction. To
assess induction, preprocessing and configuration selection must use only the
training subset, with held-out samples reserved for assignment and evaluation.
The new Nutrimouse inductive notebook illustrates that separation and checks
single-row, batch, order, frozen-state, and serialization invariants. It is an
exploratory example on a small dataset, not a clinical validation of patient
assignment quality or a reproduction of the full-data transductive scores.
