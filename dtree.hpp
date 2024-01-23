#ifndef DTREE_HPP_
#define DTREE_HPP_

#include <iostream>
#include <list>
#include <vector>

// HYPER-PARAMS
#ifndef NUM_THRESHOLDS
    #define NUM_THRESHOLDS 2
#endif
#define PRUNE_LIMIT 50
#define NUM_BINS 50

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
        features = f;
    }

    DataObject(std::vector<double> f)
    {
        data_class = f.back(); f.pop_back();
        features = std::move(f);
    }

    DataObject(std::vector<double> &f)
    {
        data_class = f.back(); f.pop_back();
        features = f;
    }

};

class DecisionTree
{
public:
    enum TrainingType
    {
        Optimized,
        Randomized,
    };

    virtual DecisionTree& train(std::list<DataObject> &, TrainingType);
    DecisionTree(size_t num_subtrees = 2) : subtrees(num_subtrees), thresholds(num_subtrees) {};

    virtual std::vector<double> predict(std::vector<double> &);
    // int predict(DataObject) = delete;
    size_t num_classes;
private:
    std::vector<DecisionTree*> subtrees;
    size_t attr_idx;
    std::vector<double> thresholds;

    std::pair<size_t, std::vector<double>> choose_attribute_optimized(std::list<DataObject> &);
    std::pair<size_t, std::vector<double>> choose_attribute_randomized(std::list<DataObject> &);
    void train_(std::list<DataObject> &, std::vector<double>&, TrainingType, DecisionTree*);

    // Some stuff only needed by special nodes
    std::vector<double> class_dist;
};

class DecisionForest : public DecisionTree
{
private:
    std::vector<DecisionTree*> trees;
public:
    DecisionForest(uint num_trees) 
    {
        trees = std::vector<DecisionTree*>(num_trees);
        for (size_t i = 0; i < num_trees; ++i) trees[i] = new DecisionTree();
    }

    DecisionForest& train(std::list<DataObject> &, DecisionForest::TrainingType);
    std::vector<double> predict(std::vector<double>&);
};

#endif