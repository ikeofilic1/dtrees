#include "dtree.hpp"
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <numeric>

template<typename T>
void print_vec(std::vector<T> v)
{
    using std::cout;
    cout << "{ ";
    for (T t : v)
        cout << t << ", ";
    cout << "}\n";
}

size_t get_num_unique_classes(std::list<DataObject> &examples)
{    
    size_t max = 0;
    for(auto e : examples)
    {
        if (e.data_class > 0 && (size_t)e.data_class > max)
            max = e.data_class;
    }
    
    return max+1;
}

DecisionTree &DecisionTree::train(std::list<DataObject> &examples, DecisionTree::TrainingType t) //TODO: use default value instead??
{
    num_classes = get_num_unique_classes(examples);
    
    auto vec = std::vector<double>(num_classes, 1.0/num_classes);
    train_(examples, vec, t, this);
    return *this;
}

std::vector<double> DecisionTree::predict(std::vector<double> &obj)
{
    if (class_dist.empty())
        throw std::logic_error("You must train the decision tree before you try to predict");

    if (subtrees.empty() || attr_idx >= obj.size()) // We reached a leaf node or passed an object that is too small
        return class_dist;

    size_t thresh_idx = std::upper_bound(thresholds.begin(), thresholds.end(), obj[attr_idx]) - thresholds.begin();
    return subtrees[thresh_idx]->predict(obj);
}

using std::vector, std::list;
vector<list<DataObject>> split_examples(std::list<DataObject> &examples, size_t attr_idx, vector<double> &thresholds)
{
    //auto s = std::unordered_set<DataObject>(mvit(examples.begin()), mvit(examples.end()));
    vector<list<DataObject>> ans(thresholds.size());
    for (size_t i = 0; i < ans.size(); ++i) ans[i] = list<DataObject>();

    for (auto &e : examples)
    {
        size_t idx = std::upper_bound(thresholds.begin(), thresholds.end(), e.features[attr_idx]) - thresholds.begin();
        ans[idx].push_back(e);
    }

    return ans;
}

vector<size_t> class_count(std::list<DataObject> &examples, size_t num_classes)
{
    //std::cout << num_classes << "\n";
    vector<size_t> ans(num_classes, 0);
    for (auto &e : examples)
    {
        ++ans[e.data_class];
    }

    return ans;
}

vector<double> classToDist(int clas, size_t class_size)
{
    if (clas < 0 || (size_t)clas >= class_size)
        throw std::logic_error("Invalid class "+std::to_string(clas));

    vector<double> ans(class_size,0.0);
    ans[clas] = 1.0;
    return ans;
}

bool sameClass(std::list<DataObject> &examples)
{
    int c = examples.front().data_class;
    for (auto &e : examples)
    {
        if (e.data_class != c) 
            return false;
    }
    return true;
}

vector<double> dist(std::list<DataObject> examples, size_t num_classes)
{
    vector<double> ans(num_classes,0.0);    
    for (auto &e : examples) ++ans[e.data_class];

    int n = examples.size();
    std::for_each(ans.begin(), ans.end(), [n](double &d) { d = d/n; });
    return ans;
}

#include <cassert>

void DecisionTree::train_(std::list<DataObject> &examples, std::vector<double> &def, DecisionTree::TrainingType t, DecisionTree *curr) 
{
    //std::cout << examples->size() << "\n";
    //print_vec(def);
    // if (examples.size())
    //     print_vec(examples.front().features);
        
    if (examples.size() <= PRUNE_LIMIT) {
        curr->class_dist = def;
        curr->subtrees.clear();
    }
    else if (sameClass(examples)) {
        curr->class_dist = classToDist(examples.front().data_class, num_classes);
        curr->subtrees.clear();
    }
    else
    {
        size_t attr;
        vector<double> thresh;
        std::pair<size_t, vector<double>> a;
        switch (t)
        {
            case Optimized:
                a = choose_attribute_optimized(examples);
                attr = a.first;
                thresh = a.second;
                break;
            case Randomized:
                a = choose_attribute_randomized(examples);
                attr = a.first;
                thresh = a.second;
                break;
            default:
                throw std::invalid_argument("Invalid training type");
        }

        auto disti = dist(examples, num_classes);
        auto new_examples = split_examples(examples, attr, thresh);
        for (size_t i = 0; i < new_examples.size(); ++i)
        {
            DecisionTree *tr = new DecisionTree(NUM_THRESHOLDS);
            train_(new_examples[i], disti, t, tr);
            curr->subtrees[i] = tr;
        }
        curr->class_dist = std::move(disti);
        curr->attr_idx = attr;
        curr->thresholds = std::move(thresh);
    }
}

double entropy(vector<size_t> class_hist, int K = -1) //k3 = k1 + k2
{
    size_t kc = std::accumulate(class_hist.begin(), class_hist.end(), 0);
    if (K == -1) K = kc;

    double H = 0;
    for (auto c : class_hist)
    {
        if (c)
        {
            H += (c*log2((double)kc/c))/K;
        }
    }

    return H;
}

double info_gain(std::list<DataObject> &examples, size_t attr_idx, vector<double> thresholds, size_t num_classes)
{
    int n = examples.size();
    auto n_examples = split_examples(examples, attr_idx, thresholds);
    double H = entropy(class_count(examples, num_classes), n);
    //std::cout << H << "=====\n";

    for (auto &e : n_examples)
    {
        H -= entropy(class_count(e, num_classes), n);
    }

    //std::cout << "===I() = " << H << "\n";
    return H;
}

// This is still binary right now. Change to n-ary in the future
std::pair<vector<double>, double> choose_thresholds(std::list<DataObject> &examples, size_t attr_idx, size_t num_classes)
{
    double mxg = -1;
    double mx = std::numeric_limits<double>::min(), mn = std::numeric_limits<double>::max();
    for (auto &e : examples)
    {
        auto c = (e.features)[attr_idx];
        if (c > mx) mx = c;
        if (c < mn) mn = c;
    }

    double best = mn;
    auto inf = std::numeric_limits<double>::has_infinity
        ? std::numeric_limits<double>::infinity()
        : HUGE_VAL;
    
    for (int i = 0; i < NUM_BINS; ++i)
    {
        double t = mn + (mx-mn)*((double)i+1)/(NUM_BINS+1);
        double gain = info_gain(examples, attr_idx, (vector<double>){t,inf}, num_classes);
        //assert(gain >= 0);
        if (gain > mxg)
        {
            mxg = gain;
            best = t;
        }
    }

    //std::cout << "Best: " << best << "   " << "Gain: " << mxg << "\n";
    return {{best, inf}, mxg};
}


std::pair<size_t, std::vector<double>> DecisionTree::choose_attribute_optimized(std::list<DataObject> &examples)
{
    size_t winner = -1;
    vector<double> t_win;
    double mxg = -1;

    size_t num_attributes = examples.front().features.size();
    for (size_t i = 0; i < num_attributes; ++i)
    {
        auto [t, gain] = choose_thresholds(examples, i, num_classes);
        if (gain > mxg)
        {
            mxg = gain;
            winner = i;
            t_win = std::move(t);
        }
    }

    // std::cout << "Attribute idx: " << winner << "\n";
    // std::cout << "Threshold " << t_win[0] << " " << t_win[1] << "\n";
    return {winner, t_win};
}
std::pair<size_t, std::vector<double>> DecisionTree::choose_attribute_randomized(std::list<DataObject> &examples)
{
    size_t idx = rand()%(examples.front().features.size());
    return {idx, choose_thresholds(examples, idx, num_classes).first};
}

DecisionForest &DecisionForest::train(std::list<DataObject> &objects, DecisionTree::TrainingType ty = Randomized)
{
    for (DecisionTree *t : trees)
    {
        t->train(objects, ty);
    }
    return *this;
}

std::vector<double> DecisionForest::predict(std::vector<double> &obj)
{
    std::vector<double> ans(trees[0]->num_classes);
    for (DecisionTree *t : trees)
    {
        std::vector<double> p_dist = t->predict(obj);
        
        for (size_t i = 0; i < p_dist.size(); ++i) 
        {
            ans[i] += p_dist[i];
        }
    }

    int n = trees.size();
    std::for_each(ans.begin(), ans.end(), [n](double &d) { d = d/n; });
    return ans;
}