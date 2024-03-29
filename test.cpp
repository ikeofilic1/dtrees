#include "dtree.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include<ctime>
#include <unordered_map>

using namespace std;

DataObject object_from_string(string &);
vector<size_t> winners_from_dist(vector<double>);

template <typename A>
vector<A> tokenize_string(string &);

template <typename T>
list<T> apply_file_by_lines(ifstream &ifs, function<T(string &)> fn);

// clang-format off
static const unordered_map<string,DecisionTree::TrainingType> enum_map =
{
    {"optimized", DecisionTree::Optimized},
    {"randomized", DecisionTree::Randomized},
};
// clang-format on

void usage(const char *prog_name)
{
    cerr << "Usage: " << prog_name << "<training_file> <test_file> <option>\n";
    cerr
        << "Where: \n"
        << "  The first argument is the name of the training file, where the training data is stored.\n"
           "  The second argument is the name of the test file, where the test data is stored.\n"
           "  The third argument can have four possible values: optimized, randomized, forest3, or forest15\n";
}

int main(int argc, char const *argv[])
{
    srand((unsigned) time(0));

    if (argc < 4)
    {
        usage(argv[0]);
        return 1;
    }
    const char *training_file = argv[1];
    const char *testing_file = argv[2];
    const char *option = argv[3];

    // Check if parameters are valid first
    ifstream trainf(training_file);
    if (!trainf)
    {
        cerr << "Uh oh, " << training_file << " could not be opened for reading!\n";
        return 1;
    }

    ifstream testf(testing_file);
    if (!testf)
    {
        cerr << "Uh oh, " << testing_file << " could not be opened for reading!\n";
        return 1;
    }

    DecisionTree *dtree;
    DecisionTree::TrainingType train_as = DecisionTree::Randomized;
    if (strcasecmp(option, "forest15") == 0) dtree = new DecisionForest(15);
    else if (strcasecmp(option, "forest3") == 0) dtree = new DecisionForest(3);
    else
    {
        bool found = false;
        {
            for (auto p : enum_map)
            {
                if (strcasecmp(p.first.c_str(), option) == 0)
                {
                    found = true;
                    train_as = p.second;
                    break;
                }
            }
            
            if (!found)
            {
                cerr << "Invalid option " << option << endl;
                return 1;
            }
            dtree = new DecisionTree;
        }
    }

    auto training_objects = apply_file_by_lines<DataObject>(trainf, object_from_string);
    dtree->train(training_objects, train_as);

    ofstream dbgf("out.txt");
    if (!dbgf)
    {
        cerr << "Could not open out.txt for writing, aborting!\n";
        return 1;
    }

    string st;
    size_t idx = 0;
    double avg_accuracy = 0.0;
    while (getline(testf, st))
    {
        vector<double> vec = tokenize_string<double>(st);

        int actual_class = vec.back();
        vec.pop_back();

        auto predicted = winners_from_dist(dtree->predict(vec));
        double accuracy = find(predicted.begin(), predicted.end(), actual_class) != predicted.end()
                              ? (1.0 / predicted.size())
                              : 0;
        avg_accuracy += accuracy;
        dbgf << "Object Index = " << idx++ << ", Result = " << predicted[0] << ", True Class = " << actual_class << ", Accuracy = " << accuracy << endl;
    }
    
    dbgf << "Classification Accuracy = " << avg_accuracy/idx << "\n";
    return 0;
}

DataObject object_from_string(string &s)
{
    return DataObject(tokenize_string<double>(s));
}

template <typename T>
list<T> apply_file_by_lines(ifstream &ifs, function<T(string &)> fn)
{
    list<T> res;

    string s;
    while (getline(ifs, s))
    {
        res.push_back(fn(s));
    }

    return res;
}

template <typename A>
vector<A> tokenize_string(string &s)
{
    stringstream ss(s);
    vector<A> res;

    A temp;
    while (ss >> temp)
    {
        res.push_back(temp);
    }

    return res;
}

vector<size_t> winners_from_dist(vector<double> v)
{
    std::vector<size_t> indices;
    double current_max = 0;

    for (size_t i = 0; i < v.size(); ++i)
    {
        if (v[i] > current_max)
        {
            current_max = v[i];
            indices.clear();
        }

        if (v[i] == current_max)
        {
            indices.push_back(i);
        }
    }

    return indices;
}