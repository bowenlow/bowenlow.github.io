---
layout: single
title: "Results for Fake Hotel Review"
date: 2018-02-18
---

# Findings
## *Supervised Learning* 
<table>
    <tr>
        <th>Classifier</th>
        <th>Precision</th>
    </tr>
    <tr>
        <td>Decision Tree</td>
        <td>0.5942</td>
    </tr>
    <tr>
        <td>Logistic Regression</td>
        <td>0.6882</td>
    </tr>
    <tr>
        <td>Gaussian Naive Bayes</td>
        <td>0.7712</td>
    </tr>
    <tr>
        <td>SVM</td>
        <td>0.7312</td>
    </tr>
    <tr>
        <td>XGBoost</td>
        <td>0.8469</td>
    </tr>
</table>

## *SemiSupervised Learning*
Several different approaches were used. 
<table>
    <tr>
        <th>Clustering</th>
        <th>Results</th>
    </tr>
    <tr>
        <td>Pseudo Labelling</td>
        <td>Slightly better than chance</td>
    </tr>
    <tr>
        <td>sklearn.LabelSpreading</td>
        <td>Recall: 0.5347, Precision: 0.6276</td>
    </tr>
    <tr>
        <td>Hierarchical Clustering</td>
        <td>Coph_Distance: 0.5580</td>
    </tr>
    <tr>
        <td>LocalitySensitiveHashing</td>
        <td>Slightly better than chance</td>
    </tr>
    <tr>
        <td>DBSCAN</td>
        <td>Slightly better than chance</td>
    </tr>
    <tr>
        <td>Recursive KMeans</td>
        <td>Recall: 0.58, Precision: 0.53
    </tr>
    <tr>
        <td>Contrastive Pessimistic Likelihood Estimation (CPLE)</td>
        <td>As good as baseline model (XGBoost)
    </tr>
    <tr>
        <td>CPLE-SVM (RBF kernel)</td>
        <td>Slightly better than chance</td>
    </tr>
</table>