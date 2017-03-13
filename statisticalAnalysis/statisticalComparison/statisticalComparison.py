import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.stats import shapiro,levene,ttest_ind,wilcoxon

from stac.nonparametric_tests import quade_test,holm_test,friedman_test
from stac.parametric_tests import anova_test,bonferroni_test
from tabulate import tabulate

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def SSbetween(accuracies):
    return float(sum(accuracies.sum(axis=1)**2))/len(accuracies[0]) - float(accuracies.sum()**2)/(len(accuracies[0])*len(accuracies))

def SSTotal(accuracies):
    sum_y_squared = sum([value**2 for value in accuracies.flatten()])
    return sum_y_squared - float(accuracies.sum() ** 2) / (len(accuracies[0]) * len(accuracies))

def eta_sqrd(accuracies):
    return SSbetween(accuracies)/SSTotal(accuracies)

def multipleAlgorithmsNonParametric(algorithms,accuracies,result,alpha=0.05):
    algorithmsDataset = {x: y for (x, y) in zip(algorithms, accuracies)}
    if len(algorithms) < 5:
        result = result +"----------------------------------------------------------\n"
        result = result +"Applying Quade test\n"
        result = result +"----------------------------------------------------------\n"
        (Fvalue, pvalue, rankings, pivots) = quade_test(*accuracies)
    else:
        result = result +"----------------------------------------------------------\n"
        result = result +"Applying Friedman test\n"
        result = result +"----------------------------------------------------------\n"
        (Fvalue, pvalue, rankings, pivots) = friedman_test(*accuracies)
    result = result + format("F-value: %f, p-value: %s\n" % (Fvalue, pvalue))
    if (pvalue < alpha):
        result = result +"Null hypothesis is rejected; hence, models have different performance\n"
        r = {x: y for (x, y) in zip(algorithms, rankings)}
        sorted_ranking = sorted(r.items(), key=operator.itemgetter(1))
        sorted_ranking.reverse()
        result = result +  tabulate(sorted_ranking, headers=['Technique', 'Ranking']) +"\n"
        (winner, _) = sorted_ranking[0]
        result = result + format("Winner model: %s\n" % winner)
        result = result +"----------------------------------------------------------\n"
        result = result +"Applying Holm p-value adjustment procedure and analysing effect size\n"
        result = result +"----------------------------------------------------------\n"
        pivots = {x: y for (x, y) in zip(algorithms, pivots)}

        (comparions, zvalues, pvalues, adjustedpvalues) = holm_test(pivots, winner)
        res = zip(comparions, zvalues, pvalues, adjustedpvalues)

        result = result + tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value'])
        for (c, p) in zip(comparions, adjustedpvalues):
            cohend = abs(cohen_d(algorithmsDataset[winner], algorithmsDataset[c[c.rfind(" ") + 1:]]))
            if (cohend <= 0.2):
                effectsize = "Small"
            elif (cohend <= 0.5):
                effectsize = "Medium"
            else:
                effectsize = "Large"
            if (p > alpha):
                #print("There are not significant differences between: %s and %s (Cohen's d=%s, %s)" % (
                #winner, c[c.rfind(" ") + 1:], cohend, effectsize))
                result = result + format("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)\n" % (
                    winner, np.mean(algorithmsDataset[winner]),
                np.std(algorithmsDataset[winner]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
            else:
                result = result + format("There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)\n" % (
                    winner, np.mean(algorithmsDataset[winner]),
                np.std(algorithmsDataset[winner]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
    else:
        result = result +"Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models\n"
        result = result +"----------------------------------------------------------\n"
        result = result +"Analysing effect size\n"
        result = result +"----------------------------------------------------------\n"
        means = np.mean(accuracies, axis=1)

        maximum = max(means)
        result = result + format("We take the model with the best mean (%s, mean: %f) and compare it with the other models: \n" % (
        algorithms[means.index(maximum)], maximum))
        for i in range(0,len(algorithms)):
            if i != means.tolist().index(maximum):
                cohend = abs(cohen_d(algorithmsDataset[algorithms[means.tolist().index(maximum)]], algorithmsDataset[algorithms[i]]))
                if (cohend <= 0.2):
                    effectsize = "Small"
                elif (cohend <= 0.5):
                    effectsize = "Medium"
                else:
                    effectsize = "Large"

                result = result + format("Comparing effect size of %s and %s: Cohen's d=%s, %s\n" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
    eta= eta_sqrd(accuracies)
    if (eta <= 0.01):
        effectsize = "Small"
    elif (eta <= 0.06):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    result = result + format("Eta squared: %f (%s)\n" % (eta,effectsize))

    return result


def multipleAlgorithmsParametric(algorithms,accuracies,result,alpha=0.05):
    algorithmsDataset = {x: y for (x, y) in zip(algorithms, accuracies)}
    result = result +"----------------------------------------------------------\n"
    result = result +"Applying ANOVA test\n"
    result = result +"----------------------------------------------------------\n"
    (Fvalue, pvalue, pivots) = anova_test(*accuracies)
    result = result + format("F-value: %f, p-value: %s\n" % (Fvalue, pvalue))
    if (pvalue < alpha):
        result = result +"Null hypothesis is rejected; hence, models have different performance\n"
        result = result +"----------------------------------------------------------\n"
        result = result +"Applying Bonferroni-Dunn post-hoc and analysing effect size\n"
        result = result +"----------------------------------------------------------\n"
        pivots = {x: y for (x, y) in zip(algorithms, pivots)}

        (comparions, zvalues, pvalues, adjustedpvalues) = bonferroni_test(pivots, len(accuracies[0]))
        res = zip(comparions, zvalues, pvalues, adjustedpvalues)

        result = result + tabulate(res, headers=['Comparison', 'Zvalue', 'p-value', 'adjusted p-value']) + "\n"

        for (c, p) in zip(comparions, adjustedpvalues):
            cohend = abs(cohen_d(algorithmsDataset[c[0:c.find(" ")]], algorithmsDataset[c[c.rfind(" ") + 1:]]))
            if (cohend <= 0.2):
                effectsize = "Small"
            elif (cohend <= 0.5):
                effectsize = "Medium"
            else:
                effectsize = "Large"
            if (p > alpha):
                result = result + format("We can't say that there is a significant difference in the performance of the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)\n" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
                #print("There are not significant differences between: %s and %s (Cohen's d=%s, %s)" % (c[0:c.find(" ")],c[c.rfind(" ") + 1:],cohend,effectsize))
            else:
                result = result + format(
                "There is a significant difference between the models: %s (mean: %f, std: %f) and %s (mean: %f, std: %f) (Cohen's d=%s, %s)\n" % (
                c[0:c.find(" ")],
                np.mean(algorithmsDataset[c[0:c.find(" ")]]),
                np.std(algorithmsDataset[c[0:c.find(" ")]]),
                c[c.rfind(" ") + 1:],
                np.mean(algorithmsDataset[c[c.rfind(" ") + 1:]]),
                np.std(algorithmsDataset[c[c.rfind(" ") + 1:]]),cohend,effectsize))
    else:
        result = result + "Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models\n"
        result = result +"----------------------------------------------------------\n"
        result = result +"Analysing effect size\n"
        result = result +"----------------------------------------------------------\n"
        means = np.mean(accuracies, axis=1)
        maximum = max(means)
        result = result + format("We take the model with the best mean (%s, mean: %f) and compare it with the other models: \n" % (algorithms[means.tolist().index(maximum)],maximum))
        for i in range(0,len(algorithms)):
            if i != means.tolist().index(maximum):
                cohend = abs(cohen_d(algorithmsDataset[algorithms[means.tolist().index(maximum)]], algorithmsDataset[algorithms[i]]))
                if (cohend <= 0.2):
                    effectsize = "Small"
                elif (cohend <= 0.5):
                    effectsize = "Medium"
                else:
                    effectsize = "Large"

                result = result+ format("Comparing effect size of %s and %s: Cohen's d=%s, %s\n" % (algorithms[means.tolist().index(maximum)],algorithms[i],cohend, effectsize))
    eta= eta_sqrd(accuracies)
    if (eta <= 0.01):
        effectsize = "Small"
    elif (eta <= 0.06):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    result = result + format("Eta squared: %f (%s)\n" % (eta,effectsize))

    return result




def twoAlgorithmsParametric(algorithms,accuracies,result,alpha):
    (t,prob)=ttest_ind(accuracies[0], accuracies[1])
    result = result + format("Students' t: t=%f, p=%f\n" % (t,prob))
    if (prob > alpha):
        result = result + format("Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s\n" % (
            algorithms[0], algorithms[1]))
    else:
        result = result + format("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f)\n" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))
    result = result +"----------------------------------------------------------\n"
    result = result +"Analysing effect size\n"
    result = result +"----------------------------------------------------------\n"
    cohend = abs(cohen_d(accuracies[0], accuracies[1]))
    if (cohend <= 0.2):
        effectsize = "Small"
    elif (cohend <= 0.5):
        effectsize = "Medium"
    else:
        effectsize = "Large"
    if (prob <= alpha):
        result = result + format("Cohen's d=%s, %s\n" % (cohend, effectsize))

    return result


def twoAlgorithmsNonParametric(algorithms,accuracies,result,alpha):
    (t,prob)=wilcoxon(accuracies[0], accuracies[1])
    result = result + format("Wilconxon: t=%f, p=%f\n" % (t,prob))
    if (prob > alpha):
        result = result + format(
        "Null hypothesis is accepted; hence, we can't say that there is a significant difference in the performance of the models: %s and %s\n" % (
            algorithms[0], algorithms[1]))
    else:
        result = result + format("Null hypothesis is rejected; hence, there are significant differences between: %s (mean: %f, std: %f) and %s (mean: %f, std: %f)\n" % (
            algorithms[0], np.mean(accuracies[0]),np.std(accuracies[0]),algorithms[1], np.mean(accuracies[1]),np.std(accuracies[1])))
    cohend = abs(cohen_d(accuracies[0], accuracies[1]))
    result = result +"----------------------------------------------------------\n"
    result = result +"Analysing effect size\n"
    result = result +"----------------------------------------------------------\n"
    if (cohend <= 0.2):
        effectsize = "Small"
    elif (cohend <= 0.5):
        effectsize = "Medium"
    else:
        effectsize = "Large"

    if (prob <= alpha):
        result = result + format("Cohen's d=%s, %s\n" % (cohend, effectsize))

    return result





def meanStdReportAndPlot(algorithms,accuracies,result, dataset):
    result = result +"**********************************************************\n"
    result = result +"Mean and std\n"
    result = result +"**********************************************************\n"
    means = np.mean(accuracies, axis=1)
    stds = np.std(accuracies, axis=1)
    for (alg, mean, std) in zip(algorithms, means, stds):
        msg = "%s: %f (%f)" % (alg, mean, std)
        print(msg)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(np.transpose(accuracies))
    ax.set_xticklabels(algorithms)
    plt.savefig("meansSTD.png")

def checkParametricConditions(accuracies,result,alpha):
    result = result + "Checking independence \n"
    result = result + "Ok\n"
    independence = True
    result = result + "Checking normality using Shapiro-Wilk's test for normality, alpha=0.05\n"
    (W, p) = shapiro(accuracies)
    result = result + format( "W: %f, p:%f\n" % (W, p))
    if p < alpha:
        result = result + "The null hypothesis (normality) is rejected\n"
        normality = False
    else:
        result = result + "The null hypothesis (normality) is accepted\n"
        normality = True
    result = result + "Checking heteroscedasticity using Levene's test, alpha=0.05\n"
    (W, p) = levene(*accuracies)
    result = result + format( "W: %f, p:%f\n" % (W, p))
    if p < alpha:
        result = result + "The null hypothesis (heteroscedasticity) is rejected\n"
        heteroscedasticity = False
    else:
        result = result + "The null hypothesis (heteroscedasticity) is accepted\n"
        heteroscedasticity = True

    parametric = independence and normality and heteroscedasticity
    return (parametric,result)


# This is the main method employed to compare a dataset where the cross validation
# process has been already carried out.
def statisticalComparison(dataset,alpha=0.05):
    df = pd.read_csv(dataset)
    algorithms = df.ix[0:,0].values

    result = ""


    if (len(algorithms)<2):
        return "It is neccessary to compare at least two algorithms"
    accuracies = df.ix[0:,1:].values
    #print(dataset)
    result = result + format("Algorithms: %s\n"%algorithms)
    result = result + "==========================================================\n"
    result = result + "Report\n"
    result = result + "==========================================================\n"
    #meanStdReportAndPlot(algorithms,accuracies,dataset)
    result = result + "**********************************************************\n"
    result = result + "Statistical tests\n"
    result = result + "**********************************************************\n"
    result = result + "----------------------------------------------------------\n"
    result = result + "Checking parametric conditions \n"
    result = result + "----------------------------------------------------------\n"
    (parametric,result) = checkParametricConditions(accuracies,result,alpha)

    if parametric:
        result = result + "Conditions for a parametric test are fulfilled\n"
        if(len(algorithms)==2):
            result = result + "----------------------------------------------------------\n"
            result = result + "Working with 2 algorithms\n"
            result = result + "----------------------------------------------------------\n"
            result =twoAlgorithmsParametric(algorithms,accuracies,result,alpha)
        else:
            result = result + "----------------------------------------------------------\n"
            result = result + "Working with more than 2 algorithms\n"
            result = result + "----------------------------------------------------------\n"
            result=multipleAlgorithmsParametric(algorithms,accuracies,result,alpha)
    else:
        result = result + "Conditions for a parametric test are not fulfilled, applying a non-parametric test\n"
        if (len(algorithms) == 2):
            result = result + "----------------------------------------------------------\n"
            result = result + "Working with 2 algorithms\n"
            result = result + "----------------------------------------------------------\n"
            result =twoAlgorithmsNonParametric(algorithms, accuracies,result,alpha)
        else:
            result = result + "----------------------------------------------------------\n"
            result = result + "Working with more than 2 algorithms\n"
            result = result + "----------------------------------------------------------\n"
            result =multipleAlgorithmsNonParametric(algorithms, accuracies,result, alpha)

    return result





"""
def compare_method(tuple):
    iteration, train_index,test_index,data,labels, clf, params, name,metric = tuple
    trainData, testData = data[train_index], data[test_index]
    trainLabels, testLabels = labels[train_index], labels[test_index]
    #print("Iteration " + str(iteration) + " of " + name)
    if params is None:
        model = clf
        model.fit(trainData, trainLabels)
    else:
        try:
            model = RandomizedSearchCV(clf, param_distributions=params, n_iter=20,random_state=84)
            model.fit(trainData, trainLabels)
        except ValueError:
            model = RandomizedSearchCV(clf, param_distributions=params, n_iter=5,random_state=84)
            model.fit(trainData, trainLabels)

    predictions = model.predict(testData)
    try:
        if (metric == 'accuracy'):
            return (name, accuracy_score(testLabels, predictions))
        elif (metric == 'recall'):
            return (name, recall_score(testLabels, predictions))
        elif (metric == 'precision'):
            return (name, precision_score(testLabels, predictions))
        elif (metric == 'f1'):
            return (name, f1_score(testLabels, predictions))
        elif (metric == 'auroc'):
            return (name, roc_auc_score(testLabels, predictions))
    except ValueError:
        print("In the multiclass problem, only accuracy can be used as metric")
        return



def compare_methods(data,labels,listAlgorithms,listParameters,listAlgorithmNames,metric='accuracy',alpha=0.5):

    if(metric!='accuracy' and metric!='recall' and metric!='precision' and metric!='f1' and metric!='auroc'):
        print("Invalid metric")
        return


    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    results = {name:[] for name in listAlgorithmNames}


    tuple = [(i,train_index,test_index,data,labels,x,y,z,metric) for i,(train_index,test_index) in enumerate(kf.split(data))
                  for (x, y, z) in zip(listAlgorithms, listParameters, listAlgorithmNames)]

    p = Pool(len(listAlgorithms))

    comparison=p.map(compare_method,tuple)
    for (name,comp) in comparison:
        results[name].append(comp)

    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('temp.csv')
    statisticalComparison('temp.csv',alpha=alpha)"""




















