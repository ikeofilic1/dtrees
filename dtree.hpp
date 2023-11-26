#ifndef DTREE_HPP_
#define DTREE_HPP_

#include <iostream>
#include <list>
#include <vector>

class DataObject
{
public:
    int data_class;
    std::vector<double> features;

public:
    DataObject() = delete;

    DataObject(std::vector<double> &&f, int c)
    {
        data_class = c;
        features = std::move(f);
    };

    DataObject(std::vector<double> f)
    {
        data_class = f.back();
        features = std::move(f);
    };
};

class DecisionTree
{
private:
    std::vector<DecisionTree *> subtrees;
    std::vector<int> thresholds;

public:
    enum TrainingType
    {
        None, // Fake type, doesn't really do anything. Only there as a default
        Optimized,
        Randomized,
        Forest3,
        Forest15
    };
    DecisionTree &train(std::list<DataObject> &, TrainingType);

    std::vector<int> predict(std::vector<double>);
    // int predict(DataObject) = delete;
};

#endif