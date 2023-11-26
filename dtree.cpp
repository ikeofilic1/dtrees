#include "dtree.hpp"

DecisionTree &DecisionTree::train(std::list<DataObject> &objects, DecisionTree::TrainingType t)
{
    return *this;
}

std::vector<int> DecisionTree::predict(std::vector<double>)
{
    return {1};
}
