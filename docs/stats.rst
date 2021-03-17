 .. _stats:

********************************************************************
Uncertainty Propagation for Common Operations (`fiducia.stats`)
********************************************************************

.. automodapi:: fiducia.stats
   :no-heading:

Derivations for propagating uncertainty for some common operations.


Weighted Summation
==================
Gien a vector :math:`X_i` with `N` elements

.. math::
    \operatorname{Var}\left(\sum_{k=1}^{N} a_i X_i\right) = \sum_{k=1}^{N} a_i^2 \operatorname{Var}(X_i) + 2 \sum_{1\leq i}\sum_{<j<n}a_i a_j \operatorname{Cov}(X_i, X_j)

A simplified version of this can be written as

.. math::
    \operatorname{Var}(ax+by) = \sigma^2_{\text{summing independent variables}} = a^2\sigma_x^2 + b^2\sigma_y^2 + 2ab\sigma_{xy}^2

Trap Rule Variance
==================

For :math:`N` steps of :math:`\Delta x_{k} = x_{k+1} - x_{k}` where :math:`f(x_k) = y_k`, where each :math:`y_k` is independent, an integral can be approximated as

.. math::
   \int_a^b f(x) dx \approx\sum_{k=1}^{N}  \frac{\Delta x_{k-1}}{2}  (y_{k-1} + y_k)

To find the variance in the general case, use the last 3 equations.

.. math::

    \operatorname{Var}\left(\sum_{k=1}^{N} \frac{\Delta x_{k-1}}{2}(y_{k-1} + y_k) \right) = \operatorname{Var}\left(\sum_{k=1}^{N}   \frac{\Delta x_{k-1}}{2} y_{k-1} +  \sum_{k=1}^{N} \frac{\Delta x_{k-1}}{2} y_{k} \right)
    
    =\operatorname{Var}\left(\frac{1}{2} \sum_{k=1}^{N} \Delta x_{k-1} y_{k-1}\right) + \operatorname{Var}\left(\frac{1}{2} \sum_{k=1}^{N} \Delta x_{k-1}  y_{k}\right) + 2\operatorname{Cov}\left(\frac{1}{2} \sum_{k=1}^{N} \Delta x_{k-1} y_{k-1}, \frac{1}{2} \sum_{k=1}^{N} \Delta x_{k-1} y_{k}\right)

The two first :math:`\operatorname{Var}` terms are simple

.. math::
    \frac{1}{4}\left(\sum_{k=1}^{N} \operatorname{Var}(\Delta x_{k-1} y_{k-1})\right) + \frac{1}{4}\left(\sum_{k=1}^{N} \operatorname{Var}(\Delta x_{k-1} y_{k})\right)
    
    = \frac{1}{4} \left(\sum_{k=1}^{N} \Delta x_{k-1}^2 \operatorname{Var}(y_{k-1}) + \sum_{k=1}^{N} \Delta x_{k-1}^2 \operatorname{Var}( y_{k})\right)
    
    = \frac{1}{4} \left( \sum_{k=1}^{N} \Delta x_{k-1}^2 \sigma_{k-1}^2  + \sum_{k=1}^{N} \Delta x_{k-1}^2 \sigma_{k}^2  \right) = \frac{1}{4} \sum_{k=1}^{N} \Delta x_{k-1}^2 (\sigma_{k-1}^2 + \sigma_{k}^2)

where :math:`\sigma_k` is the uncertainty of :math:`y_k`. Now for the Covariance of the two summations

.. math::
    \operatorname{Cov}\left(\frac{1}{2} \sum_{k=1}^{N} \Delta x_{k-1} y_{k-1}, \frac{1}{2} \sum_{k=1}^{N} \Delta x_{k-1} y_{k}\right) = \frac{1}{4} \sum_{k=1}^{N-1} \Delta x_{k-1} \Delta x_{k} \sigma_{k}^2  

Combine all this and the fourth equation to get

.. math::
    \frac{1}{4} \left(\sum_{k=1}^{N} \Delta x_{k-1}^2 (\sigma_{k-1}^2 + \sigma_{k}^2) + 2 \sum_{k=1}^{N-1} \Delta x_{k-1} \Delta x_{k} \sigma_{k}^2 \right)

In the simple case of a uniform grid, where all :math:`\Delta x_k = \Delta x` this can be expanded to

.. math::
    \frac{\Delta x^2}{4} \left(\sigma_0^2 + 4\sigma_1^2 + 4\sigma_2^2 + ... + 4\sigma_{N-1}^2 + \sigma_N^2 \right)


Gradient Variance
=================
From the `numpy.grad <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_ documentation, the gradient for discrete step sizes is approximated by

.. math::
    \nabla f(x_i) = \frac{h_{i-1}^2 f(x_i+h_i) + (h_i^2 - h_{i-1}^2) f(x_i) - h_i^2 f(x_i - h_{i-1})}{h_i h_{i-1}(h_i + h_{i-1})}

with the gradient at the first and the last data point being

.. math::
    \nabla f(x_1) = \frac{y_2 - y_1}{h_1}, \nabla f(x_N) = \frac{y_N - y_{N-1}}{h_{N-1}}

for a list of :math:`x_i` data points and :math:`h_d = x_{i+1} - x_i = h_i` and :math:`h_s = x_i - x_{i-1} = h_{i-1}`. From this we can say that :math:`f(x_i) = y_i` and :math:`f(x_i + h_i) = y_{i+1}` and :math:`f(x_i - h_s) = y_{i-1}`.
Using the variance of weighted sums where the weights are the :math:`h` terms we get

.. math::
    \operatorname{Var}(\nabla y_i) = \sigma_{\nabla y_i}^2= \frac{h_{i-1}^4 \sigma_{i+1}^2 + (h_i^2 - h_{i-1}^2)^2 \sigma_i^2 - h_i^4 \sigma_{i-1}^2}{(h_i h_{i-1} (h_i + h_{i-1}))^2} 

where :math:`\sigma_i` corresponds to the uncertainty in :math:`y_i`. Keep in mind that there are :math:`N` :math:`y` values and :math:`N-1` :math:`h` values because :math:`h_i` is the difference between the :math:`N` data points.
Note that unlike in the Trap rule integration, the covariant term is 0 because no repeated uncertainty terms appear in the sum.
At the borders :math:`i =0,N` the variance is

.. math::
    \operatorname{Var}(\nabla y_1) = \frac{\sigma_2^2 - \sigma_1^2}{h_1^2}, \operatorname{Var}(\nabla y_N) = \frac{\sigma_N^2 - \sigma_{N-1}^2}{h_{N-1}^2}

Dot Product Variance
====================
The dot product of vectors :math:`X,Y` is defined as

.. math::
    X \cdot Y = \sum_{i=1}^N x_i y_i

Uncertainty propagation when multiplying two variables, :math:`f(u,v) = a u v`, with :math:`a` as a constant, is given by

.. math::
    \operatorname{f(u,v)} = (au\sigma_v)^2 + (av\sigma_u)^2 + 2a^2 uv\sigma_{uv}^2

For the dot product we assume there is no covariance between :math:`X` and :math:`Y` because they are independent. This gets us 

.. math::
\operatorname{Var}(X \cdot Y) = \sum_{i=1}^N  (x_i \sigma_{y_i})^2 + (y_i \sigma_{x_i})^2

Linear interpolation
====================
Linear interpolation of a point :math:`x_0 < x < x_1` is given by

.. math::
    y = \frac{y_0 ( x_1 - x) + y_1 (x- x_0)}{x_1 - x_0}

First thing we can do, to make this calculation easier, is assume that there is no uncertainty in the :math:`x_i` terms. This is a short cut, but it`s all that's required for the application in which this linear interpolation is being implemented.
Starting with the numerator, we can use the weighted summation rules to say

.. math::
    \operatorname{Var}(y_0 (x_1 - x) + y_1 (x - x_0))= (x_1-x)^2 \sigma_{y_0}^2 + (x - x_0)^2 \sigma_{y_1}^2

where there is no covariant term, because we assume :math:`y_0` and :math:`y_1` and independent. Then, including the denominator as a constant (because we assume all :math:`x` values have no uncertainty, we get a variance of

.. math::
    \operatorname{Var}(y)  = \frac{1}{(x_1 - x_0)^2} ( (x_1-x)^2 \sigma_{y_0}^2 + (x - x_0)^2 \sigma_{y_1}^2 )
